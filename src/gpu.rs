use std::{borrow::Cow, collections::HashMap, fmt::Debug};

use crate::{Graph, Op, Tensor};
use include_dir::{include_dir, Dir};
use wgpu::util::DeviceExt;

static SHADER_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/shader");

pub struct GPUExecutor {
    device: Option<wgpu::Device>,
    encoder: Option<wgpu::CommandEncoder>,
    queue: Option<wgpu::Queue>,
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
    let size = match t {
        Tensor::F32 { values, .. } => values.as_ref().unwrap().len(),
        Tensor::F64 { values, .. } => values.as_ref().unwrap().len(),
    };
    Ok(size.clone())
}

impl GPUExecutor {
    pub fn new() -> Self {
        Self {
            device: None,
            encoder: None,
            queue: None,
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
                _ => return Err(format!("Tensor type {:?} is not supported ", tensor_val)),
            };

            self.storage_buf_map.insert(tensor_name.clone(), buf);
        }

        // Prepare staging buffers. There will be one staging buffer corresponding to
        // each terminal node output.
        let terminals = graph
            .op_map
            .values()
            .filter(|o| o.nexts.len() == 0)
            .map(|o| o.op_name.clone())
            .collect::<Vec<String>>();

        for terminal in &terminals {
            let t_node = &graph.op_map[terminal];
            for output in &t_node.outputs {
                let tensor = &graph.tensor_map[output];
                let staging_buf = match tensor {
                    Tensor::F32 { values, shape } => {
                        create_staging_buf(&device, &output, values, shape)
                    }
                    Tensor::F64 { values, shape } => todo!(),
                };
                self.staging_buf_map.insert(output.clone(), staging_buf);
            }
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Execute nodes in topological order
        let sorted_op_names = topo(&graph.op_map);
        for op_name in sorted_op_names {
            let op = &graph.op_map[&op_name];
            match op.op_type.as_str() {
                "Relu" | "Double" => self.simple_execution(&device, &mut encoder, op, graph)?,

                _ => {
                    return Err(format!("Op `{}` is unsupported yet", op.op_type));
                }
            }
        }

        for terminal in &terminals {
            let t_node = &graph.op_map[terminal];
            for output in &t_node.outputs {
                let output_buf = &self.storage_buf_map[output];
                let staging_buf = &self.staging_buf_map[output];

                // Copy from GPU to CPU
                encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, staging_buf.size());
            }
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let mut receiver_map = HashMap::new();
        let mut buffer_slice_map = HashMap::new();

        for terminal in &terminals {
            let t_node = &graph.op_map[terminal];
            for output in &t_node.outputs {
                let staging_buf = &self.staging_buf_map[output];
                let buffer_slice = staging_buf.slice(..);

                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                receiver_map.insert(output, receiver);
                buffer_slice_map.insert(output, buffer_slice);
            }
        }
        device.poll(wgpu::Maintain::Wait);

        for terminal in &terminals {
            let t_node = &graph.op_map[terminal];
            for output in &t_node.outputs {
                let staging_buf = &self.staging_buf_map[output];
                if let Some(Ok(())) = receiver_map[output].receive().await {
                    let data = buffer_slice_map[output].get_mapped_range();

                    let out_tensor = &graph.tensor_map[output];
                    let t = match out_tensor {
                        Tensor::F32 { values: _, shape } => Tensor::F32 {
                            values: Some(bytemuck::cast_slice(&data).to_vec()),
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
        }

        self.device = Some(device);
        self.queue = Some(queue);
        Ok(())
    }

    fn simple_execution(
        &mut self,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        op: &Op,
        graph: &Graph,
    ) -> Result<(), String> {
        let src = SHADER_DIR
            .get_file(format!("{}.wgsl", op.op_type.as_str()))
            .unwrap()
            .contents_utf8()
            .unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
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
        let bindgroup_layout = compute_pipeline.get_bind_group_layout(0);
        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bindgroup_layout,
            entries: &bindgroup_entries.as_slice(),
        });

        let len = tensor_len(&graph.tensor_map[&op.inputs[0]])?;
        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bindgroup, &[]);
            cpass.insert_debug_marker(&op.op_name);
            cpass.dispatch_workgroups(len as u32, 1, 1);
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
