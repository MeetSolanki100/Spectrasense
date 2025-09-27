from components.SpeechChatbot import SpeechChatbot

def main():
    # Initialize chatbot
    chatbot = SpeechChatbot(
        whisper_model="base",  # or "small", "medium", "large"
        llm_model="mistral:7b",  # or other Ollama models
        glasses_device="Smart Glasses"  # Name of your audio device
    )
    
    print("Speech Chatbot initialized!")
    print("Choose mode:")
    print("1. Single interaction")
    print("2. Continuous listening")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        user_input, response = chatbot.single_interaction()
    elif choice == "2":
        chatbot.listen_continuously()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()