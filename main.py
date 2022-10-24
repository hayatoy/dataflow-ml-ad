

"""A pipeline that uses RunInference API to perform object detection."""

import argparse
import io
import logging
import json
from typing import Iterable
from typing import Tuple
from typing import Mapping
from typing import Any
from typing import Dict

import apache_beam as beam
import torch
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image
from torchvision import models
from torchvision import transforms


def read_image(project_id: str, element: Dict[str, str]) -> Tuple[str, Image.Image]:
  image_file_name = f"gs://{project_id}-dataflow-ml-ad/{element['filename']}"
  with FileSystems().open(image_file_name, 'r') as file:
    image_data = Image.open(io.BytesIO(file.read())).convert('RGB')
    return element['filename'], image_data


def preprocess_image(image_data: Image.Image) -> torch.Tensor:
  # remove .cuda() if you run this pipeline without GPU
  transform = transforms.ToTensor()
  return transform(image_data).cuda()


class PostProcessor(beam.DoFn):
  LABELS = ["", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

  def process(self, element: Tuple[str, PredictionResult]) -> Iterable[Dict]:
    filename, prediction_result = element
    prediction = prediction_result.inference
    for bbox, label, score in zip(prediction['boxes'].cpu().detach().numpy(),
                                  prediction['labels'].cpu().detach().numpy(),
                                  prediction['scores'].cpu().detach().numpy()):
      yield ({'filename': filename,
              'bbox': json.dumps([float(x) for x in bbox]),
              'label_id': int(label),
              'label': self.LABELS[int(label)],
              'score': float(score)})


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_state_dict_path',
      dest='model_state_dict_path',
      required=True,
      help="Path to the model's state_dict.")
  return parser.parse_known_args(argv)


def run(
    argv=None,
    save_main_session=True) -> PipelineResult:
  """
  Args:
    argv: Command line arguments defined for this example.
    save_main_session: Used for internal testing.
  """
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  gc_options = pipeline_options.view_as(GoogleCloudOptions)

  model_class = models.detection.fasterrcnn_resnet50_fpn
  model_params = {'num_classes': 91}

  # In this example we pass keyed inputs to RunInference transform.
  # Therefore, we use KeyedModelHandler wrapper over PytorchModelHandler.
  model_handler = PytorchModelHandlerTensor(
          state_dict_path=known_args.model_state_dict_path,
          model_class=model_class,
          model_params=model_params,
          device='GPU')

  # override batch_elements_kwargs to change min/max batch size
  def batch_elements_kwargs_() -> Mapping[str, Any]:
    return {'min_batch_size':10, 'max_batch_size':10}
  model_handler.batch_elements_kwargs = batch_elements_kwargs_


  query = f"""
  SELECT
    filename
  FROM
    `{gc_options.project}.dfdemo.interpolated`
  WHERE
    frame_id = 'center_camera'
  ORDER BY
    timestamp
  """
  table_schema = "filename:STRING, bbox:STRING, label_id:NUMERIC, label:STRING, score:FLOAT64"

  pipeline = beam.Pipeline(options=pipeline_options)
  filename_value_pair = (
      pipeline
      | 'ReadImageNames' >> beam.io.ReadFromBigQuery(
        project=gc_options.project, use_standard_sql=True, query=query)
      | 'ReadImageData' >> beam.Map(
        lambda element: read_image(gc_options.project, element))
      | 'PreprocessImages' >> beam.MapTuple(
        lambda image_file_name, image_data: (image_file_name, preprocess_image(image_data))))
  predictions = (
      filename_value_pair
      | 'PyTorchRunInference' >> RunInference(KeyedModelHandler(model_handler))
      | 'ProcessOutput' >> beam.ParDo(PostProcessor()))
  predictions | "WriteToBigQuery" >> beam.io.WriteToBigQuery(
    f"{gc_options.project}:dfdemo.inference",
    project=gc_options.project,
    schema=table_schema,
    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)

  result = pipeline.run()
  return result


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()