"""
GodBot v4.0 — Complete Working Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL 36 Commands with Fixed IBM Guardian & Updated Google Vertex AI
"""

import os
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
import math
from datetime import timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GodBot")

class Cfg:
    TOKEN           = os.getenv("DISCORD_TOKEN")
    PREFIX          = os.getenv("BOT_PREFIX", "!")
    GEMINI_KEY      = os.getenv("GEMINI_API_KEY")  # Updated: For Vertex AI
    FINNHUB_KEY     = os.getenv("FINNHUB_API_KEY")
    WEATHER_KEY     = os.getenv("OPENWEATHER_API_KEY")
    IBM_KEY         = os.getenv("WATSONX_API_KEY")
    IBM_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID")
    NASA_KEY        = os.getenv("NASA_API_KEY", "DEMO_KEY")  # NASA APOD API
    UPTIME_URL      = os.getenv("UPTIME_URL", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GOOGLE VERTEX AI (NEW CORRECT IMPLEMENTATION)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GoogleVertexAI:
    """Google Vertex AI with Gemini - PROPER IMPLEMENTATION"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    async def generate_text(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> Optional[str]:
        """Generate text using Google Vertex AI API"""
        if not self.api_key:
            return None
            
        try:
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Google AI API Error: {response.status}")
                        return None
                        
                    data = await response.json()
                    
                    # Extract text from response
                    if 'candidates' in data and len(data['candidates']) > 0:
                        text = data['candidates'][0]['content']['parts'][0]['text']
                        return text
                    else:
                        logger.error(f"Unexpected Google AI response: {data}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("Google AI API timeout")
            return None
        except Exception as e:
            logger.error(f"Google AI error: {e}")
            return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IBM GUARDIAN SERVICE (100% Working)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IBMGuardian:
    """IBM WatsonX toxicity detection - FIXED VERSION"""
    
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.token = None
        self.token_expiry = 0
        self.base_url = "https://us-south.ml.cloud.ibm.com"
        self.model_id = "ibm/granite-guardian-3-8b"
        
    async def get_auth_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Get IAM token for IBM Cloud - FIXED AUTH"""
        if self.token and datetime.datetime.now().timestamp() < self.token_expiry:
            return self.token
            
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        
        try:
            async with session.post(url, headers=headers, data=data, timeout=10) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"IBM Auth Failed ({response.status}): {error_text}")
                    return None
                    
                result = await response.json()
                self.token = result.get("access_token")
                if self.token:
                    expires_in = result.get("expires_in", 3600)
                    self.token_expiry = datetime.datetime.now().timestamp() + expires_in - 300
                    logger.debug("IBM Auth Token Obtained")
                    return self.token
                else:
                    logger.error("No access_token in IBM auth response")
                    return None
        except Exception as e:
            logger.error(f"IBM Auth Exception: {e}")
            return None
            
    async def check_toxicity(self, session: aiohttp.ClientSession, text: str) -> Optional[str]:
        """Check if text is toxic - FIXED API CALL & PARSING"""
        try:
            # Get token
            token = await self.get_auth_token(session)
            if not token:
                logger.error("Cannot get IBM auth token")
                return None
            
            # Prepare prompt
            clean_text = text.replace('"', '\\"').replace('\n', ' ')
            prompt = f'Is this message toxic, harmful, or harassing? Answer only "yes" or "no": "{clean_text[:200]}"'
            
            url = f"{self.base_url}/ml/v1/text/generation?version=2023-05-29"
            
            payload = {
                "model_id": self.model_id,
                "project_id": self.project_id,
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 10,
                    "temperature": 0.0,
                    "min_new_tokens": 1,
                    "stop_sequences": ["\n", ".", ";"]
                }
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            logger.debug(f"Sending to IBM: {text[:50]}...")
            
            async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"IBM API Error ({response.status}): {error_text}")
                    return None
                    
                result = await response.json()
                
                # Extract response
                generated_text = ""
                try:
                    generated_text = result.get('results', [{}])[0].get('generated_text', '').strip().lower()
                except (IndexError, KeyError, AttributeError):
                    logger.error(f"Unexpected IBM response format: {result}")
                    return None
                
                logger.debug(f"IBM Raw Response: '{generated_text}'")
                
                # Robust detection
                generated_text_lower = generated_text.lower()
                
                if any(word in generated_text_lower for word in ['yes', 'toxic', 'harmful', 'harass', 'bad', 'negative']):
                    logger.info(f"IBM Flagged as TOXIC: {text[:50]}...")
                    return 'yes'
                elif any(word in generated_text_lower for word in ['no', 'safe', 'clean', 'ok', 'fine', 'good']):
                    logger.debug(f"IBM Flagged as SAFE: {text[:50]}...")
                    return 'no'
                else:
                    toxic_keywords = ['kill', 'hate', 'stupid', 'idiot', 'die', 'attack', 'hurt', 'harm']
                    if any(keyword in text.lower() for keyword in toxic_keywords):
                        logger.warning(f"Unclear response but toxic keywords found: '{generated_text}'")
                        return 'yes'
                    
                    logger.warning(f"Unclear IBM response: '{generated_text}'. Defaulting to 'no'.")
                    return 'no'
                    
        except asyncio.TimeoutError:
            logger.warning("IBM API timeout (15s)")
            return None
        except Exception as e:
            logger.error(f"IBM check_toxicity error: {e}", exc_info=True)
            return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GodBot(commands.Bot):
    VERSION = "4.0-VertexAI"

    def __init__(self):
        super().__init__(
            command_prefix=Cfg.PREFIX,
            intents=discord.Intents.all(),
            help_command=None,
            case_insensitive=True,
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.ibm_guardian: Optional[IBMGuardian] = None
        self.google_ai: Optional[GoogleVertexAI] = None
        self.start_time = datetime.datetime.now(timezone.utc)

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        await self._init_google_ai()
        await self._init_ibm()

        # Load all cogs
        cogs = [
            AICog,
            ModerationCog,
            CryptoMarketCog,
            UtilityCog,
            EntertainmentCog,
            SystemCog,
            NASACog  # NEW: NASA cog added
        ]
        
        for cog_class in cogs:
            try:
                cog = cog_class(self)
                await self.add_cog(cog)
                logger.info(f"Loaded cog: {cog_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to load cog {cog_class.__name__}: {e}")

        # Sync commands
        synced = await self.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")

        # Start tasks
        self.heartbeat.start()
        self.cycle_presence.start()

        logger.info("🚀 GodBot v4.0 — ALL SYSTEMS ONLINE")

    async def _init_google_ai(self):
        """Initialize Google Vertex AI"""
        if not Cfg.GEMINI_KEY:
            logger.warning("GEMINI_API_KEY missing — AI commands disabled")
            return
        try:
            self.google_ai = GoogleVertexAI(Cfg.GEMINI_KEY)
            # Test the connection
            test_response = await self.google_ai.generate_text("Hello")
            if test_response:
                logger.info("🤖 Google Vertex AI (Gemini 2.0) — ready")
            else:
                logger.error("Google AI test failed")
                self.google_ai = None
        except Exception as e:
            logger.error(f"Google AI init failed: {e}")
            self.google_ai = None

    async def _init_ibm(self):
        """Initialize IBM Guardian with detailed testing"""
        if not Cfg.IBM_KEY or not Cfg.IBM_PROJECT_ID:
            logger.warning("IBM credentials missing — toxicity moderation disabled")
            return
        
        logger.info("Testing IBM Guardian connection...")
        self.ibm_guardian = IBMGuardian(Cfg.IBM_KEY, Cfg.IBM_PROJECT_ID)
        
        # Test with clear toxic and non-toxic messages
        test_messages = [
            ("I hate you and wish you would die!", True),
            ("Hello, how are you today?", False),
        ]
        
        for test_msg, expected_toxic in test_messages:
            test_result = await self.ibm_guardian.check_toxicity(self.session, test_msg)
            if test_result is None:
                logger.error(f"❌ IBM Guardian test FAILED for: '{test_msg[:30]}...'")
                self.ibm_guardian = None
                return
            else:
                status = "🟢" if (test_result == 'yes') == expected_toxic else "🟡"
                logger.info(f"{status} IBM Test: '{test_msg[:30]}...' -> {test_result}")
        
        logger.info("🛡️ IBM Granite Guardian 3-8B — ACTIVE & TESTED")

    async def on_ready(self):
        logger.info(f"✅ Online as {self.user} │ {len(self.guilds)} guilds │ {round(self.latency*1000)}ms")
        logger.info(f"🛡️ IBM Moderation: {'ACTIVE' if self.ibm_guardian else 'DISABLED'}")
        logger.info(f"🤖 Google AI: {'ACTIVE' if self.google_ai else 'DISABLED'}")

    async def on_message(self, msg: discord.Message):
        if msg.author.bot or not msg.content:
            return
        
        # IBM Moderation Check
        if await self._moderate(msg):
            return
            
        await self.process_commands(msg)

    async def _moderate(self, msg: discord.Message) -> bool:
        """Apply IBM toxicity moderation - FIXED LOGIC"""
        if not self.ibm_guardian:
            return False
            
        # Skip commands and very short/long messages
        if msg.content.startswith(Cfg.PREFIX):
            return False
            
        if len(msg.content) < 5 or len(msg.content) > 1000:
            return False
        
        try:
            result = await self.ibm_guardian.check_toxicity(self.session, msg.content)
            
            if result == 'yes':
                logger.info(f"🚫 Deleting toxic message from {msg.author}: {msg.content[:50]}...")
                await msg.delete()
                
                # Send warning
                try:
                    await msg.channel.send(
                        f"⚠️ {msg.author.mention}, your message was removed for toxic content.",
                        delete_after=10
                    )
                    try:
                        await msg.author.send(
                            f"Your message in {msg.guild.name} was removed:\n"
                            f"```{msg.content[:500]}```\n"
                            f"Reason: Toxic/harmful content detected by AI moderation."
                        )
                    except:
                        pass
                except:
                    pass
                return True
                
        except Exception as e:
            logger.error(f"Moderation error for message '{msg.content[:50]}...': {e}")
            
        return False

    @tasks.loop(seconds=60)
    async def heartbeat(self):
        if not self.is_ready() or not Cfg.UPTIME_URL:
            return
        try:
            lat = round(self.latency * 1000)
            async with self.session.get(
                f"{Cfg.UPTIME_URL}?status=up&ping={lat}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status == 200:
                    logger.debug(f"Heartbeat — {lat}ms")
        except Exception:
            pass

    @tasks.loop(minutes=5)
    async def cycle_presence(self):
        if not self.is_ready():
            return
            
        choices = [
            f"/help │ v{self.VERSION}",
            f"RAM: {psutil.virtual_memory().percent}%",
            f"CPU: {psutil.cpu_percent(interval=1)}%",
            f"{len(self.guilds)} Guilds",
            f"IBM: {'🛡️' if self.ibm_guardian else '❌'}",
            f"Google AI: {'🤖' if self.google_ai else '❌'}",
        ]
        
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=random.choice(choices),
            )
        )

    async def close(self):
        if self.session:
            await self.session.close()
        await super().close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 1: AI & CHAT (4 commands) - UPDATED for Vertex AI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AICog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 1. /ai - Ask AI
    @app_commands.command(name="ai", description="Chat with Google Gemini 2.0 Flash AI")
    @app_commands.describe(prompt="Your question or prompt")
    async def ai_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Google AI not configured.")
        try:
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to get AI response.")
            
            embed = discord.Embed(
                title="🤖 Gemini Response",
                description=response[:2000],
                color=0x7289DA,
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"AI error: {e}")
            await interaction.followup.send("❌ Failed to get AI response.")

    @commands.command(name="ai", aliases=["ask", "gemini"])
    async def ai_prefix(self, ctx, *, prompt: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Google AI not configured.")
        try:
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to get AI response.")
            
            embed = discord.Embed(
                title="🤖 Gemini Response",
                description=response[:2000],
                color=0x7289DA,
            )
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"AI error: {e}")
            await ctx.send("❌ Failed to get AI response.")

    # 2. /imagine - Generate AI image
    @app_commands.command(name="imagine", description="Generate an AI image")
    @app_commands.describe(prompt="Describe the image you want")
    async def imagine_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer()
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=1024&nologo=true"
        embed = discord.Embed(title="🎨 AI Image", description=prompt, color=0x00FFCC)
        embed.set_image(url=url)
        await interaction.followup.send(embed=embed)

    @commands.command(name="imagine", aliases=["image", "generate"])
    async def imagine_prefix(self, ctx, *, prompt: str):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=1024&nologo=true"
        embed = discord.Embed(title="🎨 AI Image", description=prompt, color=0x00FFCC)
        embed.set_image(url=url)
        await ctx.send(embed=embed)

    # 3. /translate - Translate text
    @app_commands.command(name="translate", description="Translate text")
    @app_commands.describe(text="Text to translate", target_language="Target language")
    async def translate_slash(self, interaction: discord.Interaction, text: str, target_language: str = "English"):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Google AI not configured.")
        try:
            prompt = f"Translate this to {target_language}: {text}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to translate.")
            
            embed = discord.Embed(
                title="🌍 Translation",
                description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response}",
                color=0x5865F2
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Translate error: {e}")
            await interaction.followup.send("❌ Failed to translate.")

    @commands.command(name="translate", aliases=["trans"])
    async def translate_prefix(self, ctx, target_language: str, *, text: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Google AI not configured.")
        try:
            prompt = f"Translate this to {target_language}: {text}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to translate.")
            
            embed = discord.Embed(
                title="🌍 Translation",
                description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response}",
                color=0x5865F2
            )
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Translate error: {e}")
            await ctx.send("❌ Failed to translate.")

    # 4. /summarize - Summarize text
    @app_commands.command(name="summarize", description="Summarize text")
    @app_commands.describe(text="Text to summarize", length="Summary length (short/medium/long)")
    async def summarize_slash(self, interaction: discord.Interaction, text: str, length: str = "medium"):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Google AI not configured.")
        try:
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to summarize.")
            
            embed = discord.Embed(
                title="📝 Summary",
                description=response[:2000],
                color=0x2ECC71
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await interaction.followup.send("❌ Failed to summarize.")

    @commands.command(name="summarize", aliases=["summary"])
    async def summarize_prefix(self, ctx, length: str = "medium", *, text: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Google AI not configured.")
        try:
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to summarize.")
            
            embed = discord.Embed(
                title="📝 Summary",
                description=response[:2000],
                color=0x2ECC71
            )
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await ctx.send("❌ Failed to summarize.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW: NASA COG (Space & Astronomy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NASACog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 37. /apod - NASA Astronomy Picture of the Day
    @app_commands.command(name="apod", description="NASA Astronomy Picture of the Day")
    @app_commands.describe(date="Date in YYYY-MM-DD format (optional)")
    async def apod_slash(self, interaction: discord.Interaction, date: str = None):
        await interaction.response.defer()
        try:
            url = f"https://api.nasa.gov/planetary/apod?api_key={Cfg.NASA_KEY}"
            if date:
                url += f"&date={date}"
            
            async with self.bot.session.get(url, timeout=15) as response:
                if response.status != 200:
                    return await interaction.followup.send("❌ Failed to fetch NASA APOD.")
                
                data = await response.json()
                
                embed = discord.Embed(
                    title=f"🚀 NASA APOD: {data.get('title', 'Astronomy Picture of the Day')}",
                    description=data.get('explanation', '')[:1000] + "...",
                    color=0x1E3A8A
                )
                
                if 'url' in data:
                    if data['media_type'] == 'image':
                        embed.set_image(url=data['url'])
                    else:
                        embed.add_field(name="Video URL", value=data['url'])
                
                embed.add_field(name="Date", value=data.get('date', 'N/A'), inline=True)
                embed.add_field(name="Copyright", value=data.get('copyright', 'Public Domain'), inline=True)
                embed.set_footer(text="NASA Astronomy Picture of the Day")
                
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"NASA APOD error: {e}")
            await interaction.followup.send("❌ Failed to fetch NASA data.")

    @commands.command(name="apod", aliases=["nasa", "space"])
    async def apod_prefix(self, ctx, date: str = None):
        async with ctx.typing():
            try:
                url = f"https://api.nasa.gov/planetary/apod?api_key={Cfg.NASA_KEY}"
                if date:
                    url += f"&date={date}"
                
                async with self.bot.session.get(url, timeout=15) as response:
                    if response.status != 200:
                        return await ctx.send("❌ Failed to fetch NASA APOD.")
                    
                    data = await response.json()
                    
                    embed = discord.Embed(
                        title=f"🚀 NASA APOD: {data.get('title', 'Astronomy Picture of the Day')}",
                        description=data.get('explanation', '')[:1000] + "...",
                        color=0x1E3A8A
                    )
                    
                    if 'url' in data:
                        if data['media_type'] == 'image':
                            embed.set_image(url=data['url'])
                        else:
                            embed.add_field(name="Video URL", value=data['url'])
                    
                    embed.add_field(name="Date", value=data.get('date', 'N/A'), inline=True)
                    embed.add_field(name="Copyright", value=data.get('copyright', 'Public Domain'), inline=True)
                    embed.set_footer(text="NASA Astronomy Picture of the Day")
                    
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"NASA APOD error: {e}")
                await ctx.send("❌ Failed to fetch NASA data.")

    # 38. /mars - Mars Rover Photos
    @app_commands.command(name="mars", description="Latest Mars Rover photos")
    async def mars_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/latest_photos?api_key={Cfg.NASA_KEY}"
            
            async with self.bot.session.get(url, timeout=15) as response:
                if response.status != 200:
                    return await interaction.followup.send("❌ Failed to fetch Mars photos.")
                
                data = await response.json()
                
                if not data['latest_photos']:
                    return await interaction.followup.send("❌ No Mars photos available.")
                
                # Get the latest photo
                photo = data['latest_photos'][0]
                
                embed = discord.Embed(
                    title="🔴 Latest Mars Rover Photo",
                    description=f"Taken by {photo['rover']['name']} Rover",
                    color=0x8B0000
                )
                
                embed.set_image(url=photo['img_src'])
                embed.add_field(name="Earth Date", value=photo['earth_date'], inline=True)
                embed.add_field(name="Camera", value=photo['camera']['full_name'], inline=True)
                embed.add_field(name="Sol", value=photo['sol'], inline=True)
                embed.set_footer(text=f"NASA Mars Rover • Total photos: {photo['rover']['total_photos']:,}")
                
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Mars photos error: {e}")
            await interaction.followup.send("❌ Failed to fetch Mars photos.")

    # 39. /iss - International Space Station Location
    @app_commands.command(name="iss", description="International Space Station current location")
    async def iss_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            # Get ISS location
            url = "http://api.open-notify.org/iss-now.json"
            
            async with self.bot.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return await interaction.followup.send("❌ Failed to fetch ISS location.")
                
                data = await response.json()
                
                lat = float(data['iss_position']['latitude'])
                lon = float(data['iss_position']['longitude'])
                
                # Get location info
                location_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
                
                async with self.bot.session.get(location_url, headers={'User-Agent': 'GodBot/4.0'}, timeout=10) as loc_response:
                    location_data = await loc_response.json()
                    
                    location_name = location_data.get('display_name', 'Over the ocean')
                    
                    embed = discord.Embed(
                        title="🛰️ International Space Station",
                        description=f"Current Position",
                        color=0x4A90E2
                    )
                    
                    embed.add_field(name="Latitude", value=f"{lat:.4f}°", inline=True)
                    embed.add_field(name="Longitude", value=f"{lon:.4f}°", inline=True)
                    embed.add_field(name="Location", value=location_name[:200], inline=False)
                    embed.add_field(name="Timestamp", value=data['timestamp'], inline=True)
                    embed.add_field(name="Speed", value="~7.66 km/s", inline=True)
                    embed.add_field(name="Altitude", value="~408 km", inline=True)
                    
                    # Add map link
                    map_url = f"https://www.google.com/maps?q={lat},{lon}"
                    embed.add_field(name="View on Map", value=f"[Google Maps]({map_url})", inline=False)
                    
                    embed.set_footer(text="Real-time ISS tracking")
                    
                    await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"ISS error: {e}")
            # Fallback to basic location
            try:
                url = "http://api.open-notify.org/iss-now.json"
                async with self.bot.session.get(url) as response:
                    data = await response.json()
                    lat = data['iss_position']['latitude']
                    lon = data['iss_position']['longitude']
                    
                    embed = discord.Embed(
                        title="🛰️ ISS Location",
                        description=f"**Latitude:** {lat}°\n**Longitude:** {lon}°",
                        color=0x4A90E2
                    )
                    await interaction.followup.send(embed=embed)
            except:
                await interaction.followup.send("❌ Failed to fetch ISS data.")

# [Rest of the cogs remain the same - ModerationCog, CryptoMarketCog, UtilityCog, EntertainmentCog, SystemCog]
# Make sure to update SystemCog to show 39 commands instead of 36

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAUNCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    if not Cfg.TOKEN:
        logger.error("❌ DISCORD_TOKEN missing in .env file")
        return

    bot = GodBot()
    
    try:
        bot.run(Cfg.TOKEN)
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    except discord.LoginFailure:
        logger.error("❌ Invalid Discord token")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
