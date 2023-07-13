SAMPLE_REGION_REQUEST_DATA = {
    "tile_size": (10, 10),
    "tile_overlap": (1, 1),
    "tile_format": "NITF",
    "image_id": "test-image-id",
    "image_url": "test-image-url",
    "region_bounds": ((0, 0), (50, 50)),
    "model_name": "test-model-name",
    "model_invoke_mode": "SM_ENDPOINT",
    "output_bucket": "unit-test",
    "output_prefix": "region-request",
    "execution_role": "arn:aws:iam::012345678910:role/OversightMLBetaInvokeRole",
}

SAMPLE_IMAGE_REQUEST_DATA = {
    "imageURL": "s3://test-account/path/to/data/sample_file.tif",
    "outputBucket": "output-bucket",
    "outputPrefix": "oversight/sample",
    "modelName": "test-model",
}
