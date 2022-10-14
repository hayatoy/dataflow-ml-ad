# dataflow-ml-ad
A tutorial to deploy Dataflow ML pipeline with autonomous driving data.

## Prerequisites
- Create a [Google Cloud project](https://cloud.google.com/docs/get-started)

## Dataset Preparation
In this tutorial, we use open source dataset provided in [this repo](https://github.com/udacity/self-driving-car). This flow requires xxGB-ish disk space, therefore, do not use Cloud Shell.

1. Download [CH2_001 dataset](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2)
```sh
$ sudo apt install transmission-cli
$ transmission-cli https://github.com/udacity/self-driving-car/raw/master/datasets/CH3/CH3_001.tar.gz.torrent
$ tar -xvf CH3_001.tar.gz
```
2. Dump data from the rosbag format using [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader)
```sh
$ git clone https://github.com/rwightman/udacity-driving-reader
$ sed -i 's/bootstrap.pypa.io\/get-pip.py/bootstrap.pypa.io\/pip\/2.7\/get-pip.py/g' Dockerfile
$ ./build.sh
$ ./run-bagdump.sh -i /data -o /output
```
3. Upload image files to Google Cloud Storage
```sh
$ export PROJECT="your project id"
$ export REGION="us-central1"
$ gcloud storage buckets create gs://${PROJECT}-dataflow-ml-ad --location ${REGION}
$ gcloud storage cp -r output/center gs://${PROJECT}-dataflow-ml-ad/center
```
4. Write location data to BigQuery
```sh
$ bq --location=${REGION} mk --dataset dfdemo 
$ bq load --autodetect=true dfdemo.interpolated output/interpolated.csv
```
5. Upload Model to Google Cloud Storage
```sh
$ pip install -r requirements.txt
$ python save_model.py
$ gcloud storage cp  frcnn.pth gs://${PROJECT}-dataflow-ml-ad/
```
## Run the Pipeline

```sh
$ gcloud builds submit --config build.yaml
$ export REGION="us-central1"
$ export GPU_TYPE="nvidia-tesla-t4"
$ gcloud builds submit \
    --config run.yaml \
    --substitutions _REGION=$REGION,_GPU_TYPE=$GPU_TYPE \
    --no-source
```

## References:
- [Dataflow ML blog](https://cloud.google.com/blog/products/data-analytics/influsing-ml-models-into-production-pipelines-with-dataflow)
- [Example Dataflow GPU pipelines](https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/dataflow/gpu-examples)
- [Example RunInference pipelines](https://github.com/apache/beam/tree/master/sdks/python/apache_beam/examples/inference)
- [Udacity self driving car dataset](https://github.com/udacity/self-driving-car/tree/master/datasets)