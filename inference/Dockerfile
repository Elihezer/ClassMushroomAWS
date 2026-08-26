FROM public.ecr.aws/lambda/python:3.12

RUN pip install \
    --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ${LAMBDA_TASK_ROOT}/

RUN pip install \
    --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY HTML_page_model.py ${LAMBDA_TASK_ROOT}/
COPY online_inference.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.lambda_handler"]