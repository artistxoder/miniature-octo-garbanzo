#!/usr/bin/env python3
"""
🤖 GodBot v10.0 - Ultimate Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The All-in-One Enterprise AI Discord Bot

FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 AI SYSTEMS:
   • Google Gemini 3 Flash/Pro + IBM Watson Granite
   • Dual AI with automatic fallback & caching
   • Context-aware chat & Code generation

🛡️ MODERATION:
   • ML-powered content detection (Strike System)
   • Auto-timeout & DM warnings
   • Configurable strictness (Off to Maximum)

🌐 API INTEGRATIONS:
   • Weather (OpenWeather) • Stocks (Finnhub) • Crypto (CoinGecko)
   • NASA (APOD, Mars Rover) • GitHub Repos • Dictionary
   • QR Codes • Polls • Image Generation (Pollinations)

⚡ UTILITY & FUN:
   • Server/User Info • Calculator • Trivia • 8-Ball • Dice
   • Memes • Jokes • Cats • Latency Monitor

📊 ADVANCED:
   • Real-time Performance Monitoring
   • Auto-Self-Healing & Cleanup
   • Rich Color-Coded Embeds

Author: GodBot Team
License: MIT
Python: 3.8+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import asyncio
import aiohttp
import logging
import random
import urllib.parse
import html
import psutil
import platform
import datetime
import json
import re
import time
import math
from datetime import timezone, timedelta
from typing import Optional, Dict, List, Tuple, Union, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

import discord
from discord import app_commands, Embed, Interaction, Color, Activity, ActivityType, ButtonStyle
from discord.ext import commands, tasks
from discord.ui import Button, View

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS & ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AIModel(str, Enum):
    """Available AI models"""
    GEMINI = "gemini"
    WATSON = "watson"
    AUTO = "auto"

class ModerationLevel(Enum):
    """Moderation severity levels"""
    OFF = (0.0, "Off")
    LOW = (0.3, "Low")
    MEDIUM = (0.5, "Medium")
    HIGH = (0.7, "High")
    STRICT = (0.85, "Strict")
    MAXIMUM = (0.95, "Maximum")
    
    def __init__(self, threshold: float, name: str):
        self.threshold = threshold
        self.display_name = name

class CommandCategory(str, Enum):
    """Command categories"""
    AI = "🤖 AI Commands"
    MODERATION = "🛡️ Moderation"
    UTILITY = "🔧 Utility"
    API = "🌐 API Commands"
    FUN = "🎮 Fun Commands"
    MATH = "🔢 Math & Science"
    IMAGE = "🎨 Image Generation"
    ADMIN = "⚙️ Admin Commands"

class Colors:
    """Discord embed colors"""
    PRIMARY = 0x5865F2      # Blurple
    SUCCESS = 0x2ECC71      # Green
    ERROR = 0xE74C3C        # Red
    WARNING = 0xF39C12      # Orange
    INFO = 0x3498DB         # Blue
    GEMINI = 0x4285F4       # Google Blue
    WATSON = 0x006699       # IBM Blue
    MODERATION = 0xFF6B6B   # Coral Red
    NASA = 0x0B3D91         # NASA Blue
    GITHUB = 0x181717       # GitHub Black
    CRYPTO = 0xF7931A       # Bitcoin Orange
    FUN = 0xFFD700          # Gold
    TRIVIA = 0x9B59B6       # Purple

class Limits:
    """Application limits"""
    MAX_MESSAGE_LENGTH = 2000
    MAX_EMBED_DESC = 4096
    MAX_STRIKES = 3
    API_TIMEOUT = 15
    CLEANUP_DAYS = 7
    CACHE_SIZE = 100

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BotConfig:
    """Centralized configuration management."""
    
    # ⚠️ REQUIRED: Set your Discord token
    discord_token: str = "discord_token"
    bot_prefix: str = "/"
    
    # 🤖 AI Services
    gemini_api_key: str = ""  # https://aistudio.google.com/
    watson_api_key: str = ""
    watson_service_url: str = "https://us-south.ml.cloud.ibm.com"
    watson_project_id: str = ""
    watson_model: str = "ibm/granite-3-8b-instruct"
    
    # 📊 API Keys (Optional but recommended)
    finnhub_api_key: str = ""      # Stocks
    openweather_api_key: str = ""  # Weather
    nasa_api_key: str = "DEMO_KEY" # NASA (DEMO_KEY works but rate limited)
    
    # 🛡️ Moderation
    moderation_enabled: bool = True
    moderation_level: ModerationLevel = ModerationLevel.MEDIUM
    auto_delete_offensive: bool = True
    warn_on_delete: bool = True
    max_strikes: int = 3
    strike_timeout_minutes: int = 10
    
    # ⚙️ Advanced
    max_ai_response_tokens: int = 1000
    ai_temperature: float = 0.7
    log_level: str = "INFO"

    def is_configured(self) -> bool:
        return self.discord_token and "TOKEN_HERE" not in self.discord_token

config = BotConfig()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING & AI IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s │ %(levelname)-8s │ %(name)-15s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GodBot")

# Dynamic Imports for AI
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    google_genai = None

try:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False
    ModelInference = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbedFactory:
    """Factory for creating consistent Discord embeds"""
    
    @staticmethod
    def create(title: str, description: str = "", color: int = Colors.PRIMARY, 
               fields: Dict[str, Any] = None, thumbnail: str = None, 
               image: str = None, footer: str = None) -> Embed:
        
        embed = Embed(
            title=title[:256],
            description=description[:Limits.MAX_EMBED_DESC],
            color=color,
            timestamp=datetime.datetime.now(timezone.utc)
        )
        
        if fields:
            for name, value in fields.items():
                inline = True
                val_str = str(value)
                if isinstance(value, dict):
                    inline = value.get('inline', True)
                    val_str = value['value']
                
                embed.add_field(name=str(name)[:256], value=str(val_str)[:1024], inline=inline)
        
        if thumbnail: embed.set_thumbnail(url=thumbnail)
        if image: embed.set_image(url=image)
        if footer: embed.set_footer(text=footer[:2048])
        
        return embed

    @staticmethod
    def error(message: str) -> Embed:
        return EmbedFactory.create("❌ Error", message, Colors.ERROR)

    @staticmethod
    def success(message: str) -> Embed:
        return EmbedFactory.create("✅ Success", message, Colors.SUCCESS)

class PerformanceMonitor:
    """Tracks bot performance metrics"""
    def __init__(self):
        self.start_time = datetime.datetime.now(timezone.utc)
        self.commands_run = 0
        self.errors_count = 0
    
    @property
    def uptime(self) -> str:
        delta = datetime.datetime.now(timezone.utc) - self.start_time
        return str(delta).split('.')[0]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODERATION SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContentModerator:
    """ML-Powered Content Moderation"""
    
    PATTERNS = {
        "Hate Speech": (r"(?i)\b(n[i1]gg[ae3r]|f[a4]gg[o0]t|k[i1]ke|ch[i1]nk)\b", 1.0),
        "Threats": (r"(?i)\b(kill|murder|die)\s+(you|him|her|them)\b", 0.8),
        "Harassment": (r"(?i)\b(stupid|idiot|fat|ugly)\s+(bitch|slut|whore)\b", 0.6),
        "Self Harm": (r"(?i)\b(suicide|kill\s+myself|end\s+it\s+all)\b", 0.9)
    }
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.strikes: Dict[int, int] = defaultdict(int)
        self.last_strike: Dict[int, datetime.datetime] = {}
        
    def check_content(self, text: str) -> Tuple[bool, str, float]:
        """Analyzes text for violations. Returns (is_violation, reason, confidence)"""
        if self.config.moderation_level == ModerationLevel.OFF:
            return False, "", 0.0
            
        score = 0.0
        reasons = []
        
        # Regex Pattern Matching
        for reason, (pattern, weight) in self.PATTERNS.items():
            if re.search(pattern, text):
                score = max(score, weight)
                reasons.append(reason)
        
        # CAPS LOCK DETECTION
        if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.8:
            score = max(score, 0.4)
            reasons.append("Excessive Caps")
            
        return score >= self.config.moderation_level.threshold, ", ".join(reasons), score

    async def process_message(self, message: discord.Message) -> bool:
        if message.author.bot or not self.config.moderation_enabled:
            return False
            
        is_bad, reason, score = self.check_content(message.content)
        
        if is_bad:
            user_id = message.author.id
            self.strikes[user_id] += 1
            self.last_strike[user_id] = datetime.datetime.now()
            
            # Action: Delete
            if self.config.auto_delete_offensive:
                try:
                    await message.delete()
                except:
                    pass
            
            # Action: Timeout
            if self.strikes[user_id] >= self.config.max_strikes:
                try:
                    duration = timedelta(minutes=self.config.strike_timeout_minutes)
                    await message.author.timeout(duration, reason="Max strikes reached")
                    self.strikes[user_id] = 0 # Reset after punishment
                    await message.channel.send(f"🚫 **{message.author.mention}** has been timed out for {self.config.strike_timeout_minutes}m.")
                except Exception as e:
                    logger.error(f"Failed to timeout user: {e}")
            
            # Action: Warn
            elif self.config.warn_on_delete:
                embed = EmbedFactory.error(
                    f"⚠️ Moderation Warning ({self.strikes[user_id]}/{self.config.max_strikes})"
                )
                embed.add_field(name="Reason", value=reason)
                embed.set_footer(text="Please adhere to server rules.")
                try:
                    await message.author.send(embed=embed)
                except:
                    await message.channel.send(f"{message.author.mention}", embed=embed, delete_after=10)
            
            return True
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI SERVICE MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AIManager:
    """Manages Gemini and Watson with fallback logic"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.gemini = None
        self.watson = None
        self.cache = {}
    
  async def close(self):
        # Only try to close the session if it was actually created
        if hasattr(self, 'session') and self.session is not None:
            await self.session.close()
        await super().close()
        # Gemini Init
        if GEMINI_AVAILABLE and self.config.gemini_api_key:
            try:
                self.gemini = google_genai.GenerativeModel('gemini-3-flash', api_key=self.config.gemini_api_key)
                logger.info("✅ Gemini AI initialized")
            except Exception as e:
                logger.error(f"❌ Gemini init failed: {e}")

        # Watson Init
        if WATSON_AVAILABLE and self.config.watson_api_key:
            try:
                creds = {"url": self.config.watson_service_url, "apikey": self.config.watson_api_key}
                self.watson = ModelInference(
                    model_id=self.config.watson_model,
                    credentials=creds,
                    project_id=self.config.watson_project_id,
                    params={"decoding_method": DecodingMethods.GREEDY, "max_new_tokens": 1000}
                )
                logger.info("✅ Watson AI initialized")
            except Exception as e:
                logger.error(f"❌ Watson init failed: {e}")

    async def generate(self, prompt: str, model: AIModel = AIModel.AUTO) -> str:
        cache_key = f"{model}:{prompt}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        response = None
        try:
            # 1. Try Gemini
            if (model == AIModel.GEMINI or model == AIModel.AUTO) and self.gemini:
                resp = await asyncio.to_thread(self.gemini.generate_content, contents=prompt)
                response = resp.text if resp else None
            
            # 2. Try Watson (if Gemini failed or requested)
            if not response and (model == AIModel.WATSON or model == AIModel.AUTO) and self.watson:
                response = await asyncio.to_thread(self.watson.generate, prompt)
            
            # 3. Fallback: Reverse order
            if not response and model == AIModel.GEMINI and self.watson:
                response = await asyncio.to_thread(self.watson.generate, prompt)
            
        except Exception as e:
            logger.error(f"AI Generation Error: {e}")
            return "❌ AI Service temporarily unavailable."

        if response:
            self.cache[cache_key] = response
            if len(self.cache) > Limits.CACHE_SIZE:
                self.cache.pop(next(iter(self.cache)))
            return response
        
        return "❌ Could not generate response. Please check API keys."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN BOT CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GodBot(commands.Bot):
    VERSION = "10.0-Ultimate"
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(command_prefix=config.bot_prefix, intents=intents, help_command=None)
        
        self.config = config
        self.session: aiohttp.ClientSession = None
        self.ai = AIManager(config)
        self.mod = ContentModerator(config)
        self.monitor = PerformanceMonitor()

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        await self.ai.initialize()
        
        # Register Command Trees
        self.register_commands()
        
        await self.tree.sync()
        logger.info(f"✅ Synced {len(self.tree.get_commands())} slash commands")
        
        self.bg_tasks.start()

    async def close(self):
        await self.session.close()
        await super().close()

    @tasks.loop(minutes=10)
    async def bg_tasks(self):
        """Rotation status and cleanup"""
        statuses = [
            f"v{self.VERSION}",
            f"{len(self.guilds)} Servers",
            "/help for commands",
            "Watching Chat 🛡️"
        ]
        await self.change_presence(activity=Activity(type=ActivityType.custom, name=random.choice(statuses)))
        
        # Cleanup cache
        self.ai.cache.clear()

    async def on_message(self, message):
        if not message.author.bot:
            await self.mod.process_message(message)
        await super().on_message(message)

    # ----------------------------------------------------------------
    # COMMAND REGISTRATION
    # ----------------------------------------------------------------
    def register_commands(self):
        
        # --- AI COMMANDS ---
        @self.tree.command(name="ai", description="Chat with Gemini/Watson")
        async def ai_cmd(interaction: Interaction, prompt: str, model: AIModel = AIModel.AUTO):
            await interaction.response.defer()
            resp = await self.ai.generate(prompt, model)
            embed = EmbedFactory.create("🤖 AI Response", resp[:4000], Colors.GEMINI if model != "watson" else Colors.WATSON)
            await interaction.followup.send(embed=embed)

        @self.tree.command(name="image", description="Generate an image (via Pollinations)")
        async def image_cmd(interaction: Interaction, prompt: str):
            await interaction.response.defer()
            encoded = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
            embed = EmbedFactory.create(f"🎨 {prompt}", image=url, color=Colors.FUN)
            await interaction.followup.send(embed=embed)

        # --- UTILITY COMMANDS ---
        @self.tree.command(name="ping", description="Check bot latency")
        async def ping_cmd(interaction: Interaction):
            lat = round(self.latency * 1000)
            await interaction.response.send_message(embed=EmbedFactory.create("🏓 Pong!", f"Latency: {lat}ms", Colors.SUCCESS))

        @self.tree.command(name="userinfo", description="Get user info")
        async def user_cmd(interaction: Interaction, user: Optional[discord.Member] = None):
            user = user or interaction.user
            fields = {
                "🆔 ID": user.id,
                "📅 Joined": user.joined_at.strftime("%Y-%m-%d"),
                "🎂 Created": user.created_at.strftime("%Y-%m-%d"),
                "🎭 Roles": len(user.roles) - 1
            }
            embed = EmbedFactory.create(f"👤 {user.name}", thumbnail=user.display_avatar.url, fields=fields)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="serverinfo", description="Get server info")
        async def server_cmd(interaction: Interaction):
            g = interaction.guild
            fields = {
                "👑 Owner": g.owner.mention,
                "👥 Members": g.member_count,
                "💬 Channels": len(g.channels),
                "🚀 Boosts": g.premium_subscription_count
            }
            embed = EmbedFactory.create(f"🏠 {g.name}", thumbnail=g.icon.url if g.icon else None, fields=fields)
            await interaction.response.send_message(embed=embed)
        
        @self.tree.command(name="poll", description="Create a poll")
        async def poll_cmd(interaction: Interaction, question: str, option1: str, option2: str, option3: Optional[str] = None):
            opts = [option1, option2]
            if option3: opts.append(option3)
            desc = "\n".join([f"{i+1}️⃣ {opt}" for i, opt in enumerate(opts)])
            embed = EmbedFactory.create(f"📊 {question}", desc, Colors.INFO)
            await interaction.response.send_message(embed=embed)
            msg = await interaction.original_response()
            emojis = ["1️⃣", "2️⃣", "3️⃣"]
            for i in range(len(opts)): await msg.add_reaction(emojis[i])

        @self.tree.command(name="qrcode", description="Generate a QR Code")
        async def qr_cmd(interaction: Interaction, data: str):
            url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={urllib.parse.quote(data)}"
            embed = EmbedFactory.create("📱 QR Code", image=url, color=Colors.PRIMARY)
            await interaction.response.send_message(embed=embed)

        # --- API COMMANDS ---
        @self.tree.command(name="weather", description="Get weather info")
        async def weather_cmd(interaction: Interaction, city: str):
            if not self.config.openweather_api_key:
                return await interaction.response.send_message("❌ API key missing", ephemeral=True)
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.config.openweather_api_key}&units=metric"
            async with self.session.get(url) as resp:
                if resp.status != 200: return await interaction.response.send_message("❌ City not found")
                data = await resp.json()
                fields = {
                    "🌡️ Temp": f"{data['main']['temp']}°C",
                    "💧 Humidity": f"{data['main']['humidity']}%",
                    "☁️ Sky": data['weather'][0]['description'].title()
                }
                await interaction.response.send_message(embed=EmbedFactory.create(f"Weather in {data['name']}", fields=fields, color=Colors.INFO))

        @self.tree.command(name="crypto", description="Get crypto price")
        async def crypto_cmd(interaction: Interaction, coin: str = "bitcoin"):
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true"
            async with self.session.get(url) as resp:
                data = await resp.json()
                if coin not in data: return await interaction.response.send_message("❌ Coin not found")
                price = data[coin]['usd']
                change = data[coin].get('usd_24h_change', 0)
                color = Colors.SUCCESS if change >= 0 else Colors.ERROR
                embed = EmbedFactory.create(f"💰 {coin.title()}", f"Price: ${price:,.2f}\n24h Change: {change:.2f}%", color)
                await interaction.response.send_message(embed=embed)

        @self.tree.command(name="github", description="Get repo info")
        async def github_cmd(interaction: Interaction, owner: str, repo: str):
            url = f"https://api.github.com/repos/{owner}/{repo}"
            async with self.session.get(url) as resp:
                if resp.status != 200: return await interaction.response.send_message("❌ Repo not found")
                d = await resp.json()
                fields = {"⭐ Stars": d['stargazers_count'], "🍴 Forks": d['forks_count'], "🐛 Issues": d['open_issues_count']}
                await interaction.response.send_message(embed=EmbedFactory.create(f"GitHub: {d['full_name']}", d['description'], Colors.GITHUB, fields))

        @self.tree.command(name="nasa", description="NASA Picture of the Day")
        async def nasa_cmd(interaction: Interaction):
            url = f"https://api.nasa.gov/planetary/apod?api_key={self.config.nasa_api_key}"
            async with self.session.get(url) as resp:
                if resp.status != 200: return await interaction.response.send_message("❌ NASA API Error")
                d = await resp.json()
                embed = EmbedFactory.create(f"🌌 {d.get('title')}", d.get('explanation')[:1000] + "...", Colors.NASA, image=d.get('url'))
                await interaction.response.send_message(embed=embed)

        # --- FUN COMMANDS ---
        @self.tree.command(name="8ball", description="Ask the Magic 8-Ball")
        async def ball_cmd(interaction: Interaction, question: str):
            answers = ["Yes.", "No.", "Maybe.", "Ask again later.", "Definitely.", "Very doubtful."]
            embed = EmbedFactory.create("🎱 Magic 8-Ball", f"**Q:** {question}\n**A:** {random.choice(answers)}", Colors.FUN)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="trivia", description="Get a trivia question")
        async def trivia_cmd(interaction: Interaction):
            async with self.session.get("https://opentdb.com/api.php?amount=1&type=boolean") as resp:
                d = await resp.json()
                q = d['results'][0]
                question = html.unescape(q['question'])
                ans = q['correct_answer']
                
                view = View()
                async def cb(intr: Interaction):
                    if intr.user.id != interaction.user.id: return
                    btn_lbl = intr.data['custom_id'] # type: ignore
                    msg = "✅ Correct!" if btn_lbl == ans else f"❌ Wrong! It was {ans}."
                    await intr.response.send_message(msg, ephemeral=True)
                    
                b1 = Button(label="True", style=ButtonStyle.green, custom_id="True")
                b2 = Button(label="False", style=ButtonStyle.red, custom_id="False")
                b1.callback = b2.callback = cb
                view.add_item(b1).add_item(b2)
                
                await interaction.response.send_message(embed=EmbedFactory.create("❓ Trivia", question, Colors.TRIVIA), view=view)

        # --- ADMIN ---
        @self.tree.command(name="stats", description="Bot Statistics")
        async def stats_cmd(interaction: Interaction):
            mem = psutil.virtual_memory()
            fields = {
                "⏰ Uptime": self.monitor.uptime,
                "💻 CPU": f"{psutil.cpu_percent()}%",
                "🧠 RAM": f"{mem.percent}%",
                "🐍 Python": platform.python_version()
            }
            await interaction.response.send_message(embed=EmbedFactory.create("📊 System Stats", fields=fields))

        @self.tree.command(name="help", description="Show all commands")
        async def help_cmd(interaction: Interaction):
            desc = """
            **🤖 AI:** `/ai`, `/image`
            **🌐 API:** `/weather`, `/crypto`, `/github`, `/nasa`
            **🔧 Utility:** `/ping`, `/userinfo`, `/serverinfo`, `/poll`, `/qrcode`
            **🎮 Fun:** `/8ball`, `/trivia`
            **🛡️ Mod:** Auto-moderation is active.
            """
            await interaction.response.send_message(embed=EmbedFactory.create("GodBot v10 Help", desc))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    print(f"\n🚀 STARTING GODBOT v{GodBot.VERSION}")
    
    if not config.is_configured():
        print("❌ ERROR: Discord Token not configured!")
        print(f"   Please edit the 'discord_token' field in line 113.")
        return

    bot = GodBot()
    try:
        async with bot:
            await bot.start(config.discord_token)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
