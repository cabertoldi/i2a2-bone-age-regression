# kapmug-ocr 
FROM python:3.6

WORKDIR /app/i2a2-bone-age-regression

COPY . .
RUN pip install --upgrade pip \
    pip install --no-cache-dir -r requirements.txt

CMD python -m index
EXPOSE 3000