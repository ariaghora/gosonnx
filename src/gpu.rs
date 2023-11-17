use crate::graph::{Graph, Op, Tensor};
use crate::utils::tensor_len;
use include_dir::{include_dir, Dir};
use std::{borrow::Cow, collections::HashMap, fmt::Debug};
use wgpu::util::DeviceExt;
use wgpu::Limits;

pub static SHADER_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/shader");

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
    let mut n_items = shape.iter().fold(1, |x, y| x * y) as usize;
    // TODO: proper handling on 0-sized dims or non-zero-length shape but containing 0-length dim
    if n_items == 0 {
        n_items = 1;
    }
    let vals: Cow<'a, Vec<T>> = match values {
        Some(v) => Cow::Borrowed(v),
        None => Cow::Owned(vec![T::default(); n_items]),
    };

    // Some models provides tensors with empty data, i.e., with shape [0]. WGPU does not
    // allow zero buffer binding, so we trick it by using a "dummy" buffer binding with
    // size of 4 (minimum allowed)
    let tensor_has_data = vals.len() > 0;
    let data = if tensor_has_data {
        // We create buffer initialized with tensor's original data
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("{}.storage", buf_label).as_str()),
            contents: bytemuck::cast_slice(&vals),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    } else {
        // The dummy buffer
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{}.storage", buf_label).as_str()),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
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
                Tensor::I64 { values, shape } => {
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
                Tensor::I64 { values, shape } => {
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
                .get_file(format!("{}.glsl", op.op_type.to_string()))
                .unwrap()
                .contents_utf8()
                .unwrap();

            let (compiled, wg) = op.op_type.compile(shader_source, op, graph)?;
            self.execute_pass(&compiled, &device, &mut encoder, op, &wg)?;
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
                    Tensor::I64 { values: _, shape } => Tensor::I64 {
                        values: Some(
                            bytemuck::cast_slice(&data)[..tensor_len(out_tensor).unwrap()].to_vec(),
                        ),
                        shape: shape.to_vec(),
                    },
                    Tensor::F64 { .. } => todo!(),
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
        let mut defines = naga::FastHashMap::default();
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
                    min_binding_size: None,
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
            label: Some(&format!("bindgroup_{}_{}", op.op_name, op.op_type)),
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
        let mut limits = Limits::default();
        limits.max_storage_buffer_binding_size = 256 << 20;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features & wgpu::Features::TIMESTAMP_QUERY,
                    limits: limits,
                },
                None,
            )
            .await
            .unwrap();

        // println!(
        //     "BUF LIMIT: {}",
        //     device.l
        // );

        Ok((device, queue))
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
