use std::fmt::Display;
use std::{borrow::Cow, collections::HashMap, fmt::Debug};

use crate::graph::{ExportAttr, Graph, Op, OpType, Tensor};
use crate::ops::conv::ConvOp;
use crate::ops::flatten::FlattenOp;
use crate::ops::maxpool::MaxPoolOp;
use crate::ops::{gemm::GemmOp, Compile};
use include_dir::{include_dir, Dir};
use naga::FastHashMap;
use wgpu::util::DeviceExt;

static SHADER_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/shader");

pub struct GPUExecutor {
    pub storage_buf_map: HashMap<String, wgpu::Buffer>,
    pub staging_buf_map: HashMap<String, wgpu::Buffer>,
}

fn create_storage_buf<'a, T: bytemuck::Pod + Default + Debug>(
    device: &wgpu::Device,
    buf_label: &str,
    values: &'a Option<Vec<T>>,
    shape: &Vec<i64>,
) -> wgpu::Buffer {
    let n_items = shape.iter().fold(1, |x, y| x * y) as usize;
    let vals: Cow<'a, Vec<T>> = match values {
        Some(v) => Cow::Borrowed(v),
        None => Cow::Owned(vec![T::default(); n_items]),
    };
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("{}.storage", buf_label).as_str()),
        contents: bytemuck::cast_slice(&vals),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    data
}

fn create_staging_buf<'a, T: bytemuck::Pod + Default + Debug>(
    device: &wgpu::Device,
    buf_label: &str,
    values: &'a Option<Vec<T>>,
    shape: &Vec<i64>,
) -> wgpu::Buffer {
    let n_items = shape.iter().fold(1, |x, y| x * y) as usize;
    let vals: Cow<'a, Vec<T>> = match values {
        Some(v) => Cow::Borrowed(v),
        None => Cow::Owned(vec![T::default(); n_items]),
    };
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("{}.staging", buf_label).as_str()),
        contents: bytemuck::cast_slice(&vals),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    data
}

fn tensor_len(t: &Tensor) -> Result<usize, String> {
    let len = match t {
        Tensor::F32 { values: _, shape } => shape,
        Tensor::F64 { values: _, shape } => shape,
    }
    .iter()
    .fold(1, |x, y| x * y) as usize;
    Ok(len)
}

impl GPUExecutor {
    pub fn new() -> Self {
        Self {
            storage_buf_map: HashMap::new(),
            staging_buf_map: HashMap::new(),
        }
    }

    pub fn execute(&mut self, graph: &mut Graph) -> Result<(), String> {
        pollster::block_on(self.execute_async(graph))
    }

    async fn execute_async(&mut self, graph: &mut Graph) -> Result<(), String> {
        let (device, queue) = self.create_device().await?;

        // Prepare storage buffers
        for (tensor_name, tensor_val) in graph.tensor_map.iter() {
            let buf: wgpu::Buffer = match tensor_val {
                Tensor::F32 { values, shape } => {
                    create_storage_buf(&device, &tensor_name, values, shape)
                }
                Tensor::F64 { values, shape } => {
                    create_storage_buf(&device, &tensor_name, values, shape)
                }
            };

            self.storage_buf_map.insert(tensor_name.clone(), buf);
        }

        let terminal_outputs = graph.terminal_outputs();

        // Prepare staging buffers. There will be one staging buffer corresponding to
        // each terminal node output.
        for output in &terminal_outputs {
            let tensor = &graph.tensor_map[output];
            let staging_buf = match tensor {
                Tensor::F32 { values, shape } => {
                    create_staging_buf(&device, &output, values, shape)
                }
                Tensor::F64 { values, shape } => {
                    create_staging_buf(&device, &output, values, shape)
                }
            };
            self.staging_buf_map.insert(output.clone(), staging_buf);
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Execute nodes in topological order
        let sorted_op_names = topo(&graph.op_map);
        for op_name in sorted_op_names {
            let op = &graph.op_map[&op_name];
            let shader_source = SHADER_DIR
                .get_file(format!("{}.glsl", op.op_type))
                .unwrap()
                .contents_utf8()
                .unwrap();
            match &op.op_type {
                OpType::Gemm {
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                } => {
                    let gemm_op = GemmOp::new(*alpha, *beta, *trans_a, *trans_b);
                    let compiled = gemm_op.compile(op, shader_source, graph)?;
                    let (x, y) = gemm_op.compute_workgroup_size(op, graph);
                    self.execute_pass(&compiled, &device, &mut encoder, op, &[x, y, 1])?
                }
                OpType::Conv {
                    dilations,
                    group,
                    kernel_shape,
                    pads,
                    strides,
                } => {
                    let conv_op = ConvOp::new(
                        dilations.clone(),
                        *group,
                        kernel_shape.clone(),
                        pads.clone(),
                        strides.clone(),
                    );
                    let compiled = conv_op.compile(op, shader_source, graph)?;
                    let wg = conv_op.compute_workgroup_size(op, graph);
                    self.execute_pass(&compiled, &device, &mut encoder, op, &wg)?
                }
                OpType::Flatten { axis } => {
                    let flatten_op = FlattenOp::new(*axis);
                    let compiled = flatten_op.compile(op, shader_source, graph)?;
                    let wg = flatten_op.compute_workgroup_size(op, graph);
                    self.execute_pass(&compiled, &device, &mut encoder, op, &wg)?;
                }
                OpType::MaxPool {
                    ceil_mode,
                    kernel_shape,
                    pads,
                    strides,
                } => {
                    let maxpool_op = MaxPoolOp::new(
                        *ceil_mode,
                        kernel_shape.clone(),
                        pads.clone(),
                        strides.clone(),
                    );
                    let wg = maxpool_op.compute_workgroup_size(op, graph);
                    let compiled = maxpool_op.compile(op, shader_source, graph)?;
                    self.execute_pass(&compiled, &device, &mut encoder, op, &wg)?;
                }
                // Simple Op pass can be just executed.
                // - 1 input & 1 output buffer
                // - Input length = output length
                // - input type = output type
                OpType::Relu => {
                    let local_size_x = 256;
                    let numel = tensor_len(&graph.tensor_map[&op.inputs[0]]).unwrap();
                    let num_workgroups_x = (numel + local_size_x - 1) / local_size_x;
                    let wg = &[num_workgroups_x as u32, 1, 1];
                    let compiled = self.compile_unary_shader(&op.op_type)?;
                    self.execute_pass(&compiled, &device, &mut encoder, op, wg)?
                }

                _ => {
                    return Err(format!("Op `{:?}` is unsupported yet", op.op_type));
                }
            }
        }

        for output in &terminal_outputs {
            let output_buf = &self.storage_buf_map[output];
            let staging_buf = &self.staging_buf_map[output];

            // Copy from GPU to CPU
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, staging_buf.size());
        }

        let mut receiver_map = HashMap::new();
        let mut buffer_slice_map = HashMap::new();

        queue.submit(Some(encoder.finish()));

        for output in &terminal_outputs {
            let staging_buf = &self.staging_buf_map[output];
            let buffer_slice = staging_buf.slice(..);

            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            receiver_map.insert(output, receiver);
            buffer_slice_map.insert(output, buffer_slice);
        }
        device.poll(wgpu::Maintain::Wait);

        for output in &terminal_outputs {
            let staging_buf = &self.staging_buf_map[output];
            if let Some(Ok(())) = receiver_map[output].receive().await {
                let data = buffer_slice_map[output].get_mapped_range();

                let out_tensor = &graph.tensor_map[output];
                let t = match out_tensor {
                    Tensor::F32 { values: _, shape } => Tensor::F32 {
                        values: Some(
                            bytemuck::cast_slice(&data)[..tensor_len(out_tensor).unwrap()].to_vec(),
                        ),
                        shape: shape.to_vec(),
                    },
                    Tensor::F64 {
                        values: _,
                        shape: _,
                    } => todo!(),
                };

                drop(data);
                staging_buf.unmap();

                graph.output_tensor_map.insert(output.clone(), t);
            }
        }

        Ok(())
    }

    fn execute_pass(
        &mut self,
        shader_source: &str,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        op: &Op,
        num_work_groups: &[u32],
    ) -> Result<(), String> {
        let mut defines = FastHashMap::default();
        defines.insert("GL_EXT_debug_printf".into(), "enable".into());
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Glsl {
                shader: Cow::Borrowed(shader_source),
                stage: naga::ShaderStage::Compute,
                defines: defines,
            },
        });

        let mut bindgroup_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![];
        let mut cnt = 0;
        for _ in 0..&op.inputs.len() + &op.outputs.len() {
            bindgroup_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: cnt,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            });
            cnt += 1;
        }
        let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("bindgroup_layout_{}", op.op_type)),
            entries: bindgroup_layout_entries.as_slice(),
        });

        let mut bindgroup_entries: Vec<wgpu::BindGroupEntry> = vec![];
        let mut cnt = 0;
        for input in &op.inputs {
            bindgroup_entries.push(wgpu::BindGroupEntry {
                binding: cnt,
                resource: self.storage_buf_map[input].as_entire_binding(),
            });
            cnt += 1;
        }
        for output in &op.outputs {
            bindgroup_entries.push(wgpu::BindGroupEntry {
                binding: cnt,
                resource: self.storage_buf_map[output].as_entire_binding(),
            });
            cnt += 1;
        }

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("pipeline_layout_{}", op.op_type)),
            bind_group_layouts: &[&bindgroup_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("compute_pipeline_{}", op.op_type)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bindgroup_{}", op.op_type)),
            layout: &bindgroup_layout,
            entries: &bindgroup_entries.as_slice(),
        });

        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bindgroup, &[]);
            cpass.insert_debug_marker(&op.op_name);
            cpass.dispatch_workgroups(num_work_groups[0], num_work_groups[1], num_work_groups[2]);
        }
        Ok(())
    }

    async fn create_device(&self) -> Result<(wgpu::Device, wgpu::Queue), String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features & wgpu::Features::TIMESTAMP_QUERY,
                    limits: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        Ok((device, queue))
    }

    fn compile_op<Compilable: Compile>(&self, op: Compilable) {}

    fn compile_unary_shader<T: ExportAttr + Display>(&self, op_type: T) -> Result<String, String> {
        let base_shader_source = SHADER_DIR
            .get_file("_unary_elementwise.glsl")
            .unwrap()
            .contents_utf8()
            .unwrap();
        let unary_shader_source = SHADER_DIR
            .get_file(format!("{}.glsl", op_type))
            .unwrap()
            .contents_utf8()
            .unwrap();
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        context.insert("input_type", "float");
        context.insert("output_type", "float");

        tera.add_raw_templates(vec![
            ("_unary_elementwise", base_shader_source),
            (&op_type.to_string(), unary_shader_source),
        ])
        .map_err(|e| e.to_string())?;

        let compiled = tera
            .render(&op_type.to_string(), &mut context)
            .map_err(|e| e.to_string())?;
        Ok(compiled)
    }
}

fn topo_helper(op_map: &HashMap<String, Op>, sorted: &mut Vec<String>, root: &String) {
    // skip visited node
    if sorted.contains(root) {
        return;
    }

    if let Some(op) = op_map.get(root) {
        for o in &op.prevs {
            topo_helper(op_map, sorted, o);
        }
        sorted.push(root.clone())
    }
}

pub fn topo(op_map: &HashMap<String, Op>) -> Vec<String> {
    let terminals = op_map
        .values()
        .filter(|o| o.nexts.len() == 0)
        .map(|o| o.op_name.clone())
        .collect::<Vec<String>>();

    let mut sorted: Vec<String> = vec![];
    for t in terminals {
        topo_helper(op_map, &mut sorted, &t);
    }
    sorted
}
