from kfp import dsl
from mlrun.platforms import auto_mount
import os
import sys
import mlrun

funcs = {}

# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(auto_mount())


def kfpipeline():

    # Fetch the data
    ingest = funcs['fetch_data'].as_step(
        inputs={'dataset': 's3://testbucket-igz-temp/cancer-dataset.csv'},
        outputs=['dataset','return'])
    
    ingest_1 = funcs['fetch_data'].as_step(
        inputs={'dataset': ingest.outputs['dataset']},
        outputs=['dataset'])
    
    ingest_2 = funcs['fetch_data'].as_step(
        inputs={'dataset': ingest_1.outputs['dataset']},
        outputs=['dataset'])

    # Train the model
    train = funcs["trainer"].as_step(
        inputs={"dataset": ingest_2.outputs['dataset']},
        outputs=['model'])


    # Deploy the model
    deploy = funcs["serving"].deploy_step(tag=ingest.outputs['return'],models=[{'key':'cancer-classifier','model_path':train.outputs["model"], 'class_name':'mlrun.frameworks.sklearn.SklearnModelServer'}])
