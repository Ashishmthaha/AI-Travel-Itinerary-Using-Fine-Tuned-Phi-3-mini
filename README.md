# LetsGo AI Travel Planner

A fine-tuned Large Language Model (LLM)-based travel itinerary generator that creates day-wise travel plans for selected destinations.

The project fine-tunes Microsoft's Phi-3 Mini 4K Instruct model using a travel itinerary dataset and Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA). The resulting model is integrated into a Gradio-based web application for generating personalized day-wise itineraries.

## Project Overview

Traditional travel planning requires manually collecting destinations, activities, and schedules from multiple sources. This project explores the use of a fine-tuned language model to automatically generate structured travel itineraries from a destination and number of days.

The system accepts a destination and trip duration and generates a day-wise itinerary containing titles and activities.

The model is trained to produce itinerary data in JSON format, which is then processed and displayed through the application interface.

## Features

* Fine-tuned Phi-3 Mini 4K Instruct language model
* Day-wise travel itinerary generation
* Support for 2 to 5 day trips
* Destination selection through a searchable dropdown
* Structured itinerary generation
* JSON-based model output
* JSON parsing and output-repair logic
* Gradio web interface
* Hugging Face Hub model deployment
* Reusable inference setup without retraining

## Technology Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Hub
* PEFT
* LoRA
* QLoRA
* bitsandbytes
* TRL
* Gradio
* Google Colab
* CUDA

## Model

The project uses:

**Base Model:** `microsoft/Phi-3-mini-4k-instruct`

The base model is loaded using 4-bit quantization and adapted using LoRA.

The model contains approximately 3.83 billion parameters, while approximately 8.9 million parameters are made trainable during fine-tuning.

This corresponds to approximately 0.23% of the total model parameters being trained.

## Dataset

The training dataset is provided in JSON Lines (`.jsonl`) format.

The dataset contains 504 samples. Each sample contains:

* `instruction`
* `input`
* `output`

The instruction used for the training task is:

```text
Generate a detailed day-wise travel itinerary in JSON format.
```

An example input contains a destination and number of days, while the output contains a structured itinerary.

The dataset was validated to ensure that the `output` field contained valid JSON.

Dataset statistics:

| Property           | Value |
| ------------------ | ----: |
| Total samples      |   504 |
| Valid samples      |   504 |
| Training samples   |   453 |
| Evaluation samples |    51 |
| Test split         |   10% |
| Random seed        |    42 |

The dataset was split into training and evaluation sets using a 90:10 split.

## Training Data Formatting

The dataset was formatted using the Phi-3 chat template:

```text
<|user|>
Generate a detailed day-wise travel itinerary in JSON format.
Destination: Ella
Number of days: 2
<|end|>
<|assistant|>
{"destination":"Ella","total_days":2,"itinerary":[...]}
<|end|>
```

This allows the fine-tuning data to follow the conversational format expected by the Phi-3 Instruct model.

## Fine-Tuning Approach

The project uses Quantized Low-Rank Adaptation (QLoRA).

The base Phi-3 Mini model is loaded using 4-bit quantization with:

* NormalFloat 4-bit (`NF4`) quantization
* Double quantization
* Float16 computation

LoRA is then applied to selected attention and feed-forward projection layers.

### LoRA Configuration

```text
Rank (r): 16
LoRA alpha: 32
LoRA dropout: 0.05
Bias: none
Task: CAUSAL_LM
```

The LoRA adapters are applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The resulting model has approximately 8.9 million trainable parameters out of approximately 3.83 billion total parameters.

## Training Configuration

The model was fine-tuned using the Supervised Fine-Tuning (SFT) trainer from TRL.

Key training parameters:

```text
Epochs: 2
Batch size: 1
Gradient accumulation steps: 4
Learning rate: 2e-4
Optimizer: AdamW
Maximum sequence length: 1024
Gradient checkpointing: Enabled
Maximum gradient norm: 0.3
Warmup steps: 30
```

The training dataset contained 453 samples.

Training completed successfully for two epochs.

## Training Results

The training loss decreased substantially during training.

Selected training-loss values:

| Step | Training Loss |
| ---: | ------------: |
|    5 |        1.6020 |
|   30 |        0.8095 |
|   60 |        0.7733 |
|  100 |        0.5756 |
|  120 |        0.4692 |
|  150 |        0.5827 |
|  180 |        0.5598 |
|  200 |        0.5389 |
|  225 |        0.5805 |

The lowest logged training loss was approximately **0.4692 at step 120**. The final logged loss at step 225 was approximately **0.5805**.

These values represent training loss; they should not be interpreted as an independent measure of real-world itinerary quality or model accuracy.

## Inference Pipeline

After fine-tuning, the LoRA adapter is merged with the base Phi-3 Mini model.

The merged model is then used for inference.

The inference pipeline is:

```text
User selects destination
        |
        v
User selects number of days
        |
        v
Prompt construction
        |
        v
Fine-tuned Phi-3 Mini
        |
        v
Generated JSON response
        |
        v
JSON extraction and validation
        |
        v
JSON repair / cleanup
        |
        v
Day-wise itinerary
        |
        v
Gradio interface
```

## JSON Output Processing

The model is trained to generate structured JSON, but generated language-model output can sometimes contain malformed JSON.

To handle this, the application includes post-processing functions that:

* Extract the JSON portion of the response
* Attempt to parse the generated output
* Remove certain unwanted fields
* Correct missing brackets and braces
* Repair incomplete structures
* Normalize itinerary day information
* Add missing days when necessary
* Limit activities displayed for each day

This processing layer allows the application to convert imperfect model output into a more consistent itinerary format.

The implementation includes functions such as:

```text
fix_json()
pad_days()
fix_days()
try_parse()
generate_itinerary_from_model()
format_output()
```

## Web Application

The application is built using Gradio.

Users can:

1. Select a destination city.
2. Select the number of days.
3. Generate an itinerary.
4. View the resulting day-wise travel plan.
5. Copy the generated output.

The application supports trips from **2 to 5 days**.

A searchable destination list containing cities and travel destinations is included in the interface.

Example destinations include:

```text
Kochi
Goa
Tokyo
Dubai
Manali
Jaipur
Bali
Seoul
Paris
Singapore
Udaipur
Istanbul
```

## Example

Input:

```text
Destination: Kyoto
Number of Days: 3
```

Example generated structure:

```json
{
  "destination": "Kyoto",
  "total_days": 3,
  "itinerary": [
    {
      "day": 1,
      "title": "Historic Temples",
      "activities": [
        "Activity 1",
        "Activity 2",
        "Activity 3"
      ]
    }
  ]
}
```

The implemented test successfully generated an itinerary for Kyoto with three days and three activities per day after the application's processing logic.

## Deployment

The application was tested through Gradio in Google Colab using a public share link.

The model was also uploaded to the Hugging Face Hub as:

```text
ashishmthaha/letsgo-travel-phi3
```

The uploaded model was successfully loaded directly from the Hugging Face Hub after deployment.

## Reusing the Trained Model

The trained model can be loaded directly from the Hugging Face Hub without repeating the fine-tuning process.

```python
MODEL_ID = "ashishmthaha/letsgo-travel-phi3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
```

The project also includes a reusable launcher configuration for future sessions.

## Hardware and Environment

The fine-tuning process was performed in Google Colab using a NVIDIA Tesla T4 GPU.

Environment information recorded during training:

```text
GPU: NVIDIA Tesla T4
VRAM: 15.64 GB
CUDA: 12.8
```

The model was loaded in 4-bit mode during fine-tuning to reduce GPU memory requirements.

## Project Structure

A suggested repository structure is:

```text
letsgo-travel/
│
├── notebook/
│   └── fine_tuning.ipynb
│
├── app/
│   └── app.py
│
├── dataset/
│   └── dataset.jsonl
│
├── requirements.txt
└── README.md
```

The exact repository structure depends on which files are uploaded to the repository.

## Installation

Install the required Python packages:

```bash
pip install transformers==4.44.0
pip install peft==0.12.0
pip install accelerate==0.34.0
pip install trl==0.9.6
pip install datasets
pip install gradio==4.44.1
pip install bitsandbytes
pip install sentencepiece
```

A CUDA-enabled environment is required for GPU-based model inference and fine-tuning.

## Key Concepts Demonstrated

This project demonstrates practical experience with:

* Large Language Models (LLMs)
* Natural Language Processing (NLP)
* Supervised Fine-Tuning (SFT)
* Parameter-Efficient Fine-Tuning (PEFT)
* Low-Rank Adaptation (LoRA)
* Quantized Low-Rank Adaptation (QLoRA)
* 4-bit quantization
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Hub
* PyTorch
* Structured JSON generation
* Model inference
* Gradio application development
* GPU-based model training

## Limitations

The current implementation is primarily a language-model-based itinerary generator.

The generated itinerary is based on the learned training data and does not currently demonstrate integration with live external travel services such as:

* Real-time maps
* Live weather
* Hotel availability
* Flight availability
* Real-time traffic information

The initial direct inference test also demonstrated that model-generated JSON can occasionally be malformed. The application therefore includes a JSON repair and normalization layer to improve the usability of generated responses.

## Future Improvements

Possible extensions include:

* Integration with real-time map and location APIs
* Weather-aware itinerary generation
* Hotel and transportation recommendations
* Budget-based itinerary planning
* User preference-based recommendations
* Travel distance and route optimization
* Multi-language itinerary generation
* Improved structured-output reliability
* Automated evaluation of itinerary quality

## Project Information

**Project:** LetsGo AI Travel Planner

**Model:** Microsoft Phi-3 Mini 4K Instruct

**Fine-Tuning:** QLoRA / LoRA

**Interface:** Gradio

**Model Hosting:** Hugging Face Hub

**Environment:** Google Colab with NVIDIA Tesla T4
