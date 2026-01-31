# 🤖 GodBot v4.0 - Complete Discord Bot

**A feature-rich Discord bot with 39 commands, Google Vertex AI, IBM toxicity moderation, NASA space data, crypto tracking, and entertainment features.**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Discord.py](https://img.shields.io/badge/discord.py-2.3+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Commands](https://img.shields.io/badge/commands-39-orange.svg)
![Version](https://img.shields.io/badge/version-4.0-brightgreen.svg)
![NASA](https://img.shields.io/badge/NASA-API-red.svg)

## 🚀 Features

### 🤖 **AI & Chat (4 Commands)**
- `/ai` - Chat with Google Gemini 2.0 Flash via Vertex AI
- `/imagine` - Generate AI images with Pollinations.ai
- `/translate` - Translate text between languages
- `/summarize` - Summarize long text documents

### 🚀 **NASA & Space (3 NEW Commands)**
- `/apod` - NASA Astronomy Picture of the Day
- `/mars` - Latest Mars Rover photos from Curiosity
- `/iss` - Real-time International Space Station tracking

### 🛡️ **Moderation (6 Commands)**
- `/clear` - Bulk delete messages (1-100)
- `/kick` - Kick members with reason logging
- `/ban` - Ban members with message deletion
- `/timeout` - Temporary mute members
- `/warn` - Issue warnings to members
- `/userinfo` - View detailed user information

### 💰 **Crypto & Market (5 Commands)**
- `/crypto` - Real-time cryptocurrency prices (CoinGecko)
- `/stock` - Stock market data (Finnhub API)
- `/trending` - Trending cryptocurrencies
- `/exchanges` - Top crypto exchanges
- `/gas` - Ethereum gas price tracker

### 🔧 **Utility (8 Commands)**
- `/weather` - Weather forecasts worldwide
- `/timer` - Set countdown timers
- `/poll` - Create interactive polls
- `/remind` - Set personal reminders
- `/calc` - Advanced calculator
- `/define` - Dictionary definitions
- `/qr` - Generate QR codes
- `/serverinfo` - Server statistics

### 🎮 **Entertainment (5 Commands)**
- `/meme` - Random memes from Reddit
- `/joke` - Programming & general jokes
- `/8ball` - Magic 8-ball fortune teller
- `/roll` - Dice roller (2d6, 1d20, etc.)
- `/fact` - Random interesting facts

### ⚙️ **System (8 Commands)**
- `/help` - Complete command list
- `/ping` - Bot latency check
- `/stats` - Bot performance statistics
- `/uptime` - Bot uptime tracker
- `/invite` - Bot invitation link
- `/support` - Support server link
- `/vote` - Voting links
- `/about` - Bot information

## 🔧 **IBM Guardian Auto-Moderation**
**100% Working toxicity detection:**
- Automatic message scanning
- Real-time toxicity detection using IBM Granite Guardian 3-8B
- Toxic messages automatically deleted
- User warnings sent via DM
- Configurable sensitivity levels

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Discord Bot Token

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/godbot.git
cd godbot
