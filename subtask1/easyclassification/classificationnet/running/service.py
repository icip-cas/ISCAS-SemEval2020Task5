#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse

from allennlp.service.server_simple import make_app
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from classificationnet.running.utils import load_predictor
from classificationnet.utils import env_utils


def main():
    parser = argparse.ArgumentParser()
    env_utils.add_inference_argument(parser)
    parser.add_argument('-title', dest='title', type=str, default='SpanNet')
    parser.add_argument('-port', dest='port', type=int, default=5000)
    parser.add_argument('-log-file', dest='log_file', type=str)
    args = parser.parse_args()
    env_utils.pre_logger(args.log_file)

    predictor = load_predictor(model_path=args.model, device=args.device)

    app = make_app(predictor=predictor,
                   field_names=['text'],
                   title=args.title)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()


if __name__ == "__main__":
    main()
