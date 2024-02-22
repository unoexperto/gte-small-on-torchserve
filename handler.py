import logging
import os

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class ModelHandler(BaseHandler):

    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing

        """

        properties = context.system_properties
        self.manifest = context.manifest

        logger.info(f'Properties: {properties}')
        logger.info(f'Manifest: {self.manifest}')

        model_dir = properties.get("model_dir")

        # use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f'Using device {self.device}')

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = AutoModel.from_pretrained(model_dir)
            logger.info(f'Successfully loaded model {type(self.model)} from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is not None:
            logger.info('Successfully loaded tokenizer')
        else:
            raise RuntimeError('Missing tokenizer')

        # load mapping file
        # mapping_file_path = os.path.join(model_dir, 'index_to_name.json')
        # if os.path.isfile(mapping_file_path):
        #     with open(mapping_file_path) as f:
        #         self.mapping = json.load(f)
        #     logger.info('Successfully loaded mapping file')
        # else:
        #     logger.warning('Mapping file is not detected')

        self.initialized = True

    def preprocess(self, requests):
        """Tokenize the input text using the suitable tokenizer and convert
        it to tensor

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]
        """

        # unpack the data
        logger.info(f'REQ BODY: {requests}')

        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')

        texts = data.get('input')
        logger.info(f'Received {len(texts)} texts. Begin tokenizing')

        # tokenize the texts
        tokenized_data = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        logger.info(f'Tokenization process completed: {tokenized_data}')

        return tokenized_data

    def average_pool(self, tensor: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = tensor.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def inference(self, inputs, *args, **kwargs):
        """Predict class using the model

        Args:
            inputs: tensor of tokenized data
        """

        # logger.info(f'MODEL: {type(self.model)}')
        # logger.info(f'INPUTS: {type(inputs)} {inputs}')

        xxx = inputs.to(self.device)

        outputs = self.model(**inputs)

        logger.info(f'Predictions successfully created: {type(outputs)} ${outputs}')

        embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (Optionally) normalize embeddings

        logger.info('Predictions successfully created.')

        return embeddings

    def postprocess(self, outputs: Tensor):
        """
        Convert the output to the string label provided in the label mapper (index_to_name.json)

        Args:
            outputs (list): The integer label produced by the model

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        logger.info(f'Predictions are of type {type(outputs)}.')

        return [{"embeddings": outputs.tolist()}]
