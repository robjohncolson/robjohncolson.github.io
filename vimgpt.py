import openai
import pynvim
import os

# NeoVim connection
@pynvim.plugin
class ChatGPTPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim

    @pynvim.command('ChatGPTQuery', nargs='*', range='')
    def chatgpt_query(self, args, range):
        query = ' '.join(args)
        if not query:
            self.nvim.out_write("No query provided for ChatGPT.\n")
            return

        self.nvim.out_write(f"Querying ChatGPT: {query}\n")

        response = self.query_chatgpt(query)
        self.nvim.out_write(f"ChatGPT Response: {response}\n")

    def query_chatgpt(self, query):
        # Make sure to replace 'your-openai-api-key' with your actual API key
        openai.api_key = os.getenv("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")

        try:
            # Query ChatGPT
            response = openai.Completion.create(
                engine="text-davinci-003", # You can change to any model, such as gpt-3.5-turbo
                prompt=query,
                max_tokens=150
            )
            # Extract and return the text response
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"


