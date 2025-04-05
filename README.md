# ai-vs-human-images

This project allows you to upload one or more images and predicts whether each image is **Real (0)** or **AI-Generated (1)** using a trained deep learning model.

## Features

- Upload multiple images at once
- Classify images as Real or AI-Generated
- Visual display of uploaded images with predictions
- Modern UI using basic HTML and CSS
- Dockerized for easy deployment

## How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App
uvicorn main:app --reload


## Docker Instructions

You can also pull the pre-built Docker image for easier deployment:

1.  Pull the Docker image:
    `docker pull s4m0707/ai-vs-human-images:v1.0`
2.  Run the Docker container:
    `docker run -p 8000:8000 s4m0707/ai-vs-human-images:v1.0`
3.  Access the application via `http://localhost:8000`.

## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch with your feature or fix.
3.  Commit your changes.
4.  Submit a pull request for review.

## License

This project is licensed under the MIT License.