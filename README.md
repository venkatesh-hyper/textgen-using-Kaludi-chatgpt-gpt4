<b>ML CARRIER - I </B>

model https://huggingface.co/Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum

<I>gpt_model.ipynb is a python notebook file - for references</I>

<u>install requirements :-</u>

        pip install -r req.txt
 
hear we implement model from transformers and implemnt in MAIN FILE <B>gpt_model.py</B>
steps to apply
    1.import AutoTokenizer, AutoModelForSeq2SeqLM from transformers and torch also
    2.load the model and tokeniser

            def load_model():
    try:
        # Use the specified model and tokenizer from Hugging Face with TensorFlow weights
        print("Loading model with TensorFlow weights: Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
        tokenizer = AutoTokenizer.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum", from_tf=True)
        
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

  3.function to generate test 

      def generate_text(prompt, max_length=50):
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        return "Model loading failed. Check the logs for more details."
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=1000)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)

  4. summarise the output

         return tokenizer.decode(outputs[0], skip_special_tokens=True)
     
you can access the pre trained model by running stremalit file which is <u>app.py</u> which contains all the UI based materials using stremalit 
use this comment to run the model in local browser

         streamlit run app.py 

and final 

<b>we can finetune our model by using your parameters and owm dataset by using <B>train.py</B>
we need to replace your dataset in <I>train.csv</I> and <I>validation.csv</I> and run the model</b>



Train Loss: 1.2214
Validation Loss: 2.7584
Epoch: 4
