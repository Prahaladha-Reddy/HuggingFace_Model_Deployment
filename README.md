# HuggingFace_Model_Deployment
## Models Fine-Tuned

The following Hugging Face models have been fine-tuned as part of this project:

| Model Name                                   | Task Type            |
|-----------------------------------------------|----------------------|
| Prahaladha/pose_classification               | Image Classification |
| Prahaladha/disaster_management_classification | Text Classification  |
| Prahaladha/Indian_Food_Classification        | Image Classification |

![image1](image1)

## Prerequisites

- Python 3.7+
- AWS account and credentials configured (e.g., using AWS CLI)
- [Hugging Face Transformers](https://huggingface.co/transformers/) library
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for AWS integration
- Any additional dependencies in `requirements.txt`

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prahaladha-Reddy/HuggingFace_Model_Deployment.git
   cd HuggingFace_Model_Deployment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials:**
   ```bash
   aws configure
   ```

4. **Run the deployment script:**
   ```bash
   python your_script.py
   ```
