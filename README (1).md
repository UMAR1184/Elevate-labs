# Enhanced Chatbot 🤖

An intelligent conversational AI with context awareness, sentiment analysis, and personalized responses.

## ✨ Key Features

- **Context Memory**: Remembers conversation history and user preferences
- **Sentiment Analysis**: Detects emotions and adapts responses accordingly
- **Topic Detection**: Identifies and responds to hobbies, work, food, travel topics
- **Personalization**: Uses your name for tailored interactions
- **Smart Pattern Matching**: Advanced regex for better input understanding
- **Conversation Saving**: Export chat history to JSON files

## 🚀 Quick Start

```python
from Chatbot import EnhancedChatbot

bot = EnhancedChatbot()
bot.chat()
```

## 💬 Special Commands

- `summary` - View conversation overview
- `save` - Save conversation to file
- `quit/exit/bye` - End chat

## 🎯 What Makes It Smart

- Remembers your name and uses it naturally
- Detects if you're happy, sad, or tired and responds appropriately
- Suggests conversation topics when chat gets quiet
- Provides current time and date
- Tracks conversation flow and context

## 🔧 Customization

Add new topics easily:
```python
bot.topics['music'] = {
    'keywords': ['song', 'music', 'band'],
    'responses': ['Music is amazing! What genre do you enjoy?']
}
```

## 📁 Files

- `Chatbot.py` - Main chatbot code
- `requirements.txt` - Dependencies (Python standard library only)
- `chat_history_*.json` - Saved conversations

## 🎮 Run Demo

```bash
python Chatbot.py
```

Try saying: "Hello! My name is [Your Name]", "I'm feeling happy today", "What's the time?", or "summary"