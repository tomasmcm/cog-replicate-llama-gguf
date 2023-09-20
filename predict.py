from cog import BasePredictor, Input, Path
from llama_cpp import Llama

class Predictor(BasePredictor):
    def setup(self):
        """prepare the model"""
        model_path = './models/carl-llama-2-13b.Q4_K_M.gguf'
        self.llm = Llama(model_path, n_ctx=2048)

    def predict(
            self,
            prompt: str = Input(description=f"Text prompt to send to the model."),
            max_length: int = Input(
                description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
                ge=1,
                default=128
            ),
            temperature: float = Input(
                description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.8 is a good starting value.",
                ge=0.01,
                le=5,
                default=0.8,
            ),
            top_p: float = Input(
                description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
                ge=0.01,
                le=1.0,
                default=0.95
            ),
            repetition_penalty: float = Input(
                description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
                ge=0.01,
                le=5,
                default=1
            )) -> str:
        """Run a single prediction on the model"""
        # system_prompt = """This is a conversation with your Therapist AI, Carl. Carl is designed to help you while in stress. It can answer your questions and help you to calm down.\n\nContext\nYou are Carl, A Therapist AI\n"""
        # prompt_prefix = """USER: """
        # prompt_suffix = """\nCARL: """
        # prompt = system_prompt + prompt_prefix + prompt + prompt_suffix
        output = self.llm(prompt, max_tokens=max_length, temperature=temperature, top_p=top_p, repeat_penalty=repetition_penalty)
        return output["choices"][0]["text"]
