import tensorflow as tf
from .model import SavedModel

class Chatbot:
    def __init__(self, model):
        self.model = model

    def respond(self, prompt):
        prediction = self.model.predict([prompt])
        return prediction[0]

def main():
    # Load the saved model
    model = SavedModel('saved_model.pb')

    # Create a chatbot instance
    chatbot = Chatbot(model)

    # Start the chatbot loop
    while True:
        # Get the user's prompt
        prompt = input('> ')

        # Generate a response
        response = chatbot.respond(prompt)

        # Print the response
        print(response)

if __name__ == '__main__':
    main()