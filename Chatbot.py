import random
import re
import json
from datetime import datetime
from typing import Dict, List, Optional

class EnhancedChatbot:
    def __init__(self):
        self.name = "ChatBot"
        self.user_name = ""
        self.conversation_history = []
        self.user_preferences = {}
        self.context = {
            'last_topic': None,
            'mood': 'neutral',
            'conversation_count': 0
        }
        
        # Enhanced pattern matching with regex
        self.patterns = {
            'greeting': re.compile(r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b', re.IGNORECASE),
            'farewell': re.compile(r'\b(bye|goodbye|see\s+you|farewell|exit|quit)\b', re.IGNORECASE),
            'how_are_you': re.compile(r'\b(how\s+are\s+you|how\s+do\s+you\s+do|what\'?s\s+up|how\s+are\s+things)\b', re.IGNORECASE),
            'name_query': re.compile(r'\b(what\s+is\s+your\s+name|what\'?s\s+your\s+name|who\s+are\s+you)\b', re.IGNORECASE),
            'user_name': re.compile(r'\b(my\s+name\s+is|i\s+am|call\s+me)\s+(\w+)', re.IGNORECASE),
            'help': re.compile(r'\b(help|assist|support|what\s+can\s+you\s+do)\b', re.IGNORECASE),
            'weather': re.compile(r'\b(weather|temperature|rain|sunny|cloudy|forecast)\b', re.IGNORECASE),
            'time': re.compile(r'\b(time|date|day|today|clock)\b', re.IGNORECASE),
            'compliment': re.compile(r'\b(good|great|awesome|cool|nice|smart|helpful|amazing)\b.*\b(you|bot|chatbot)\b', re.IGNORECASE),
            'emotion_sad': re.compile(r'\b(sad|unhappy|depressed|down|upset|disappointed)\b', re.IGNORECASE),
            'emotion_happy': re.compile(r'\b(happy|excited|great|wonderful|fantastic|amazing|thrilled)\b', re.IGNORECASE),
            'emotion_tired': re.compile(r'\b(tired|sleepy|exhausted|weary)\b', re.IGNORECASE),
            'question': re.compile(r'.*\?$'),
            'thanks': re.compile(r'\b(thank|thanks|appreciate|grateful)\b', re.IGNORECASE),
        }
        
        # Predefined responses for different categories
        self.greetings = [
            "Hello! How can I help you today?",
            "Hi there! What's on your mind?",
            "Hey! Nice to meet you!",
            "Greetings! How are you doing?"
        ]
        
        self.farewells = [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! It was nice chatting with you!",
            "Farewell! Come back anytime!"
        ]
        
        self.how_are_you_responses = [
            "I'm doing great! Thanks for asking. How about you?",
            "I'm functioning perfectly! How are you feeling?",
            "All systems running smoothly! What about you?",
            "I'm here and ready to chat! How's your day going?"
        ]
        
        self.name_responses = [
            f"Nice to meet you! I'm {self.name}, your friendly chatbot.",
            f"My name is {self.name}. What's your name?",
            f"I'm {self.name}! Pleased to make your acquaintance.",
            f"You can call me {self.name}. I'm here to help!"
        ]
        
        self.help_responses = [
            "I can chat with you about various topics, answer simple questions, or just have a friendly conversation!",
            "I'm here to help! You can ask me questions, tell me about your day, or just chat.",
            "I can assist you with basic conversations, provide simple information, or just be a friendly companion!",
            "Feel free to ask me anything! I can discuss topics, help with simple questions, or just chat casually."
        ]
        
        self.weather_responses = [
            "I don't have access to real-time weather data, but I hope it's nice where you are!",
            "I can't check the weather, but you could try a weather app or website for current conditions.",
            "Weather talk! I wish I could tell you the forecast, but I don't have that capability.",
            "I'm not connected to weather services, but I hope you're having pleasant weather!"
        ]
        
        self.default_responses = [
            "That's interesting! Can you tell me more?",
            "I'm not sure I understand completely. Could you rephrase that?",
            "Hmm, that's something to think about. What do you think?",
            "I see! What else would you like to talk about?",
            "That's a good point! Tell me more about your thoughts on this.",
            "I'm still learning! Can you help me understand better?"
        ]
        
        # Topic-based responses for better conversation flow
        self.topics = {
            'hobbies': {
                'keywords': ['hobby', 'hobbies', 'interest', 'like to do', 'enjoy', 'fun'],
                'responses': [
                    "Hobbies are great! What do you like to do in your free time?",
                    "That sounds interesting! How did you get into that hobby?",
                    "I'd love to hear more about your interests!"
                ]
            },
            'work': {
                'keywords': ['work', 'job', 'career', 'office', 'boss', 'colleague'],
                'responses': [
                    "Work can be quite a topic! How's your job going?",
                    "That sounds like an interesting work situation. Tell me more!",
                    "Work-life balance is important. How do you manage it?"
                ]
            },
            'food': {
                'keywords': ['food', 'eat', 'hungry', 'restaurant', 'cooking', 'recipe'],
                'responses': [
                    "Food is one of life's great pleasures! What's your favorite cuisine?",
                    "That sounds delicious! Do you enjoy cooking?",
                    "I wish I could taste food! What did you have for your last meal?"
                ]
            },
            'travel': {
                'keywords': ['travel', 'trip', 'vacation', 'visit', 'country', 'city'],
                'responses': [
                    "Travel opens up so many possibilities! Where would you like to go?",
                    "That sounds like an amazing place! Have you been there before?",
                    "I'd love to hear about your travel experiences!"
                ]
            }
        }
        
        # Conversation starters for when the chat gets quiet
        self.conversation_starters = [
            "What's been the highlight of your day so far?",
            "Do you have any interesting hobbies or activities you enjoy?",
            "What's something you're looking forward to?",
            "Tell me about something that made you smile recently.",
            "What's your favorite way to spend a weekend?",
            "Is there anything new you've learned lately?"
        ]

    def preprocess_input(self, user_input: str) -> str:
        """Clean and prepare user input for processing"""
        return user_input.lower().strip()
    
    def save_to_history(self, user_input: str, bot_response: str):
        """Save conversation to history for context awareness"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'bot': bot_response
        })
        self.context['conversation_count'] += 1
        
        # Keep only last 10 exchanges to prevent memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def detect_topic(self, user_input: str) -> Optional[str]:
        """Detect the topic of conversation based on keywords"""
        processed_input = self.preprocess_input(user_input)
        
        for topic, data in self.topics.items():
            if any(keyword in processed_input for keyword in data['keywords']):
                return topic
        return None
    
    def get_contextual_response(self, topic: str) -> str:
        """Get a response based on detected topic"""
        if topic in self.topics:
            return random.choice(self.topics[topic]['responses'])
        return random.choice(self.default_responses)
    
    def analyze_sentiment(self, user_input: str) -> str:
        """Simple sentiment analysis to adjust bot responses"""
        positive_words = ['good', 'great', 'awesome', 'happy', 'love', 'like', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'hate', 'sad', 'angry', 'frustrated', 'awful', 'horrible']
        
        processed_input = self.preprocess_input(user_input)
        
        positive_count = sum(1 for word in positive_words if word in processed_input)
        negative_count = sum(1 for word in negative_words if word in processed_input)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_personalized_response(self, base_response: str) -> str:
        """Add personalization to responses when user name is known"""
        if self.user_name and random.random() < 0.3:  # 30% chance to use name
            return f"{base_response.rstrip('.')} {self.user_name}!"
        return base_response

    def get_response(self, user_input: str) -> str:
        """Generate enhanced response based on user input using pattern matching and context"""
        processed_input = self.preprocess_input(user_input)
        
        # Analyze sentiment for context
        sentiment = self.analyze_sentiment(user_input)
        self.context['mood'] = sentiment
        
        # Detect topic for better responses
        topic = self.detect_topic(user_input)
        if topic:
            self.context['last_topic'] = topic
        
        # Pattern matching with regex for more accurate detection
        
        # Greeting patterns
        if self.patterns['greeting'].search(user_input):
            response = random.choice(self.greetings)
            if self.context['conversation_count'] > 0:
                response = f"Hello again! {response.split('!', 1)[1].strip() if '!' in response else response}"
            return self.get_personalized_response(response)
        
        # Farewell patterns
        elif self.patterns['farewell'].search(user_input):
            return random.choice(self.farewells)
        
        # How are you patterns
        elif self.patterns['how_are_you'].search(user_input):
            response = random.choice(self.how_are_you_responses)
            return self.get_personalized_response(response)
        
        # Name-related queries
        elif self.patterns['name_query'].search(user_input):
            return random.choice(self.name_responses)
        
        # User sharing their name
        elif self.patterns['user_name'].search(user_input):
            match = self.patterns['user_name'].search(user_input)
            if match:
                name = match.group(2).capitalize()
                # Check if it's actually a name and not an emotion/state
                emotion_words = ['good', 'bad', 'fine', 'okay', 'sad', 'happy', 'tired', 'great', 'terrible']
                if name.lower() not in emotion_words:
                    self.user_name = name
                    self.user_preferences['name'] = name
                    return f"Nice to meet you, {self.user_name}! How can I help you today?"
            return "Nice to meet you! What would you like to chat about?"
        
        # Help requests
        elif self.patterns['help'].search(user_input):
            return random.choice(self.help_responses)
        
        # Weather queries
        elif self.patterns['weather'].search(user_input):
            return random.choice(self.weather_responses)
        
        # Time queries
        elif self.patterns['time'].search(user_input):
            current_time = datetime.now().strftime("%I:%M %p")
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"I can tell you that it's currently {current_time} on {current_date}. How can I help you today?"
        
        # Compliments to the bot
        elif self.patterns['compliment'].search(user_input):
            responses = [
                "Thank you! That's very kind of you to say. I'm just trying to be helpful!",
                "I appreciate the compliment! It makes me happy to know I'm being useful.",
                "That's so nice of you to say! I'm here whenever you need to chat."
            ]
            return self.get_personalized_response(random.choice(responses))
        
        # Questions about AI/chatbots
        elif any(word in processed_input for word in ['robot', 'ai', 'artificial', 'computer', 'machine']):
            return "Yes, I'm a chatbot! I'm an enhanced conversational AI designed to have meaningful conversations with people like you."
        
        # Emotional expressions with sentiment-aware responses
        elif self.patterns['emotion_sad'].search(user_input):
            responses = [
                "I'm sorry to hear you're feeling down. Sometimes talking can help. What's on your mind?",
                "That sounds tough. Would you like to talk about what's bothering you?",
                "I'm here to listen if you want to share what's making you feel this way."
            ]
            return self.get_personalized_response(random.choice(responses))
        
        elif self.patterns['emotion_happy'].search(user_input):
            responses = [
                "That's wonderful! I'm glad to hear you're feeling positive. What's making you happy?",
                "It's great to hear such enthusiasm! Tell me more about what's going well.",
                "Your positive energy is contagious! What's bringing you joy today?"
            ]
            return self.get_personalized_response(random.choice(responses))
        
        elif self.patterns['emotion_tired'].search(user_input):
            responses = [
                "It sounds like you might need some rest. Make sure to take care of yourself!",
                "Being tired can be tough. Have you been getting enough sleep lately?",
                "Rest is important! What's been keeping you busy?"
            ]
            return self.get_personalized_response(random.choice(responses))
        
        # Age questions
        elif 'how old' in processed_input or 'age' in processed_input:
            return "I don't have an age in the traditional sense - I'm just a computer program! How old are you, if you don't mind me asking?"
        
        # Favorite things
        elif 'favorite' in processed_input or 'favourite' in processed_input:
            return "I don't have personal preferences since I'm a chatbot, but I'd love to hear about your favorites! What are you passionate about?"
        
        # Thank you
        elif self.patterns['thanks'].search(user_input):
            responses = [
                "You're welcome! I'm happy to help. Is there anything else you'd like to talk about?",
                "My pleasure! That's what I'm here for. What else can we discuss?",
                "Glad I could help! Feel free to ask me anything else."
            ]
            return self.get_personalized_response(random.choice(responses))
        
        # Yes/No responses with context awareness
        elif processed_input in ['yes', 'yeah', 'yep', 'sure', 'absolutely']:
            if self.context['last_topic']:
                return f"Great! Let's continue talking about {self.context['last_topic']}. What would you like to know?"
            return "Great! What would you like to discuss?"
        
        elif processed_input in ['no', 'nah', 'nope', 'not really']:
            return "Okay, no problem! " + random.choice(self.conversation_starters)
        
        # Questions (ending with ?)
        elif self.patterns['question'].search(user_input):
            if topic:
                return self.get_contextual_response(topic)
            return "That's a great question! " + random.choice(self.default_responses)
        
        # Topic-based responses
        elif topic:
            return self.get_contextual_response(topic)
        
        # Default response with conversation starters for better flow
        else:
            if self.context['conversation_count'] > 3 and random.random() < 0.4:
                return random.choice(self.conversation_starters)
            return random.choice(self.default_responses)

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.conversation_history:
            return "We haven't chatted much yet!"
        
        total_exchanges = len(self.conversation_history)
        topics_discussed = set()
        
        for exchange in self.conversation_history:
            topic = self.detect_topic(exchange['user'])
            if topic:
                topics_discussed.add(topic)
        
        summary = f"We've had {total_exchanges} exchanges"
        if self.user_name:
            summary += f", {self.user_name}"
        
        if topics_discussed:
            summary += f". We've talked about: {', '.join(topics_discussed)}"
        
        return summary + "."
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
        
        conversation_data = {
            'user_name': self.user_name,
            'conversation_history': self.conversation_history,
            'user_preferences': self.user_preferences,
            'summary': self.get_conversation_summary()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            return f"Conversation saved to {filename}"
        except Exception as e:
            return f"Error saving conversation: {str(e)}"
    
    def chat(self):
        """Enhanced main chat loop with better user experience"""
        print("=" * 60)
        print(f"🤖 Welcome to {self.name} - Enhanced Edition!")
        print("I'm here to have meaningful conversations with you.")
        print("=" * 60)
        print("💡 Tips:")
        print("  • Tell me your name for a more personal experience")
        print("  • Ask me about my capabilities")
        print("  • Type 'summary' to see our conversation overview")
        print("  • Type 'save' to save our conversation")
        print("  • Type 'quit', 'exit', or 'bye' to end")
        print("=" * 60)
        
        # Start with a friendly greeting
        print(f"\n{self.name}: {random.choice(self.greetings)}")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    print(f"{self.name}: I'm here when you're ready to chat!")
                    continue
                
                # Special commands
                if user_input.lower() == 'summary':
                    print(f"{self.name}: {self.get_conversation_summary()}")
                    continue
                
                if user_input.lower() == 'save':
                    result = self.save_conversation()
                    print(f"{self.name}: {result}")
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['quit', 'exit']) or user_input.lower() in ['bye', 'goodbye']:
                    farewell = random.choice(self.farewells)
                    if self.user_name:
                        farewell = farewell.replace("!", f" {self.user_name}!")
                    print(f"{self.name}: {farewell}")
                    
                    # Offer to save conversation
                    if self.conversation_history:
                        save_prompt = input("Would you like to save our conversation? (y/n): ").strip().lower()
                        if save_prompt in ['y', 'yes']:
                            result = self.save_conversation()
                            print(f"{self.name}: {result}")
                    break
                
                # Get and display response
                response = self.get_response(user_input)
                print(f"🤖 {self.name}: {response}")
                
                # Save to history
                self.save_to_history(user_input, response)
                
            except KeyboardInterrupt:
                print(f"\n{self.name}: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"{self.name}: Sorry, I encountered an error. Let's keep chatting!")
                print(f"Debug info: {str(e)}")

def demo_chatbot():
    """Demonstrate the enhanced chatbot functionality"""
    print("🚀 Starting Enhanced Chatbot Demo...")
    print("\n✨ New Features:")
    print("  • Context-aware conversations")
    print("  • Sentiment analysis")
    print("  • Topic detection")
    print("  • Conversation history")
    print("  • Personalized responses")
    print("  • Better pattern matching")
    print("  • Conversation saving")
    
    print("\n💬 Try these sample inputs:")
    print("  • Hello! / Hi there!")
    print("  • My name is [Your Name]")
    print("  • How are you feeling today?")
    print("  • I love cooking and traveling")
    print("  • I'm feeling a bit sad today")
    print("  • What's the time?")
    print("  • You're really helpful!")
    print("  • Tell me about your capabilities")
    print("  • summary (to see conversation overview)")
    print("  • save (to save the conversation)")
    print("  • Goodbye")
    
    print("\n" + "=" * 60)
    
    # Create and start the enhanced chatbot
    bot = EnhancedChatbot()
    bot.chat()

def create_custom_chatbot(name: str = "CustomBot", additional_responses: Dict = None):
    """Factory function to create a customized chatbot"""
    bot = EnhancedChatbot()
    bot.name = name
    
    if additional_responses:
        for category, responses in additional_responses.items():
            if hasattr(bot, category):
                getattr(bot, category).extend(responses)
    
    return bot

# Example of creating a specialized chatbot
def create_support_chatbot():
    """Create a customer support focused chatbot"""
    additional_responses = {
        'help_responses': [
            "I'm here to help with your questions! What specific issue can I assist you with?",
            "Let me guide you through this. What problem are you experiencing?",
            "I'm your support assistant. How can I make your day better?"
        ]
    }
    
    support_bot = create_custom_chatbot("SupportBot", additional_responses)
    support_bot.greetings = [
        "Hello! Welcome to our support chat. How can I help you today?",
        "Hi there! I'm here to assist you with any questions or issues.",
        "Welcome! What can I help you with today?"
    ]
    
    return support_bot

if __name__ == "__main__":
    demo_chatbot()