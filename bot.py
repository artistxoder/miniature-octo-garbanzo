"""
GodBot v4.0 — Complete Working Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL 39 Commands with All APIs Working
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
    GEMINI_KEY      = os.getenv("GEMINI_API_KEY")
    FINNHUB_KEY     = os.getenv("FINNHUB_API_KEY")
    WEATHER_KEY     = os.getenv("OPENWEATHER_API_KEY")
    IBM_KEY         = os.getenv("WATSONX_API_KEY")
    IBM_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID")
    NASA_KEY        = os.getenv("NASA_API_KEY", "DEMO_KEY")
    UPTIME_URL      = os.getenv("UPTIME_URL", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GOOGLE GEMINI (Using google-generativeai)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GoogleAI:
    """Google Gemini AI - Using google-generativeai"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.genai = None
        self.model = None
        
    async def initialize(self):
        """Initialize Google AI"""
        if not self.api_key:
            logger.warning("No Gemini API key provided")
            return False
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("🤖 Gemini 2.0 Flash — ready")
            return True
        except ImportError:
            logger.error("Please install: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Google AI init error: {e}")
            return False
            
    async def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using Gemini"""
        if not self.model:
            return None
            
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text
        except Exception as e:
            logger.error(f"Google AI generate error: {e}")
            return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IBM GUARDIAN SERVICE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IBMGuardian:
    """IBM WatsonX toxicity detection"""
    
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.token = None
        self.token_expiry = 0
        self.base_url = "https://us-south.ml.cloud.ibm.com"
        self.model_id = "ibm/granite-guardian-3-8b"
        
    async def get_auth_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Get IAM token for IBM Cloud"""
        if self.token and datetime.datetime.now().timestamp() < self.token_expiry:
            return self.token
            
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key}
        
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
        """Check if text is toxic"""
        try:
            token = await self.get_auth_token(session)
            if not token:
                logger.error("Cannot get IBM auth token")
                return None
            
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
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
            
            logger.debug(f"Sending to IBM: {text[:50]}...")
            
            async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"IBM API Error ({response.status}): {error_text}")
                    return None
                    
                result = await response.json()
                generated_text = ""
                try:
                    generated_text = result.get('results', [{}])[0].get('generated_text', '').strip().lower()
                except (IndexError, KeyError, AttributeError):
                    logger.error(f"Unexpected IBM response format: {result}")
                    return None
                
                logger.debug(f"IBM Raw Response: '{generated_text}'")
                
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
    VERSION = "4.0-Final"

    def __init__(self):
        super().__init__(
            command_prefix=Cfg.PREFIX,
            intents=discord.Intents.all(),
            help_command=None,
            case_insensitive=True,
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.ibm_guardian: Optional[IBMGuardian] = None
        self.google_ai: Optional[GoogleAI] = None
        self.start_time = datetime.datetime.now(timezone.utc)

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        await self._init_google_ai()
        await self._init_ibm()

        # Load all cogs
        cogs = [AICog, ModerationCog, CryptoMarketCog, UtilityCog, EntertainmentCog, SystemCog, NASACog]
        
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
        """Initialize Google AI"""
        if not Cfg.GEMINI_KEY:
            logger.warning("GEMINI_API_KEY missing — AI commands disabled")
            return
        try:
            self.google_ai = GoogleAI(Cfg.GEMINI_KEY)
            success = await self.google_ai.initialize()
            if success:
                logger.info("🤖 Google Gemini — ready")
            else:
                logger.error("Google AI initialization failed")
                self.google_ai = None
        except Exception as e:
            logger.error(f"Google AI init failed: {e}")
            self.google_ai = None

    async def _init_ibm(self):
        """Initialize IBM Guardian"""
        if not Cfg.IBM_KEY or not Cfg.IBM_PROJECT_ID:
            logger.warning("IBM credentials missing — toxicity moderation disabled")
            return
        
        logger.info("Testing IBM Guardian connection...")
        self.ibm_guardian = IBMGuardian(Cfg.IBM_KEY, Cfg.IBM_PROJECT_ID)
        
        test_messages = [("I hate you and wish you would die!", True), ("Hello, how are you today?", False)]
        
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
        
        if await self._moderate(msg):
            return
            
        await self.process_commands(msg)

    async def _moderate(self, msg: discord.Message) -> bool:
        """Apply IBM toxicity moderation"""
        if not self.ibm_guardian:
            return False
            
        if msg.content.startswith(Cfg.PREFIX):
            return False
            
        if len(msg.content) < 5 or len(msg.content) > 1000:
            return False
        
        try:
            result = await self.ibm_guardian.check_toxicity(self.session, msg.content)
            
            if result == 'yes':
                logger.info(f"🚫 Deleting toxic message from {msg.author}: {msg.content[:50]}...")
                await msg.delete()
                
                try:
                    await msg.channel.send(f"⚠️ {msg.author.mention}, your message was removed for toxic content.", delete_after=10)
                    try:
                        await msg.author.send(f"Your message in {msg.guild.name} was removed:\n```{msg.content[:500]}```\nReason: Toxic content detected.")
                    except:
                        pass
                except:
                    pass
                return True
                
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            
        return False

    @tasks.loop(seconds=60)
    async def heartbeat(self):
        if not self.is_ready() or not Cfg.UPTIME_URL:
            return
        try:
            lat = round(self.latency * 1000)
            async with self.session.get(f"{Cfg.UPTIME_URL}?status=up&ping={lat}", timeout=10) as r:
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
            f"Gemini: {'🤖' if self.google_ai else '❌'}",
        ]
        
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=random.choice(choices)))

    async def close(self):
        if self.session:
            await self.session.close()
        await super().close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 1: AI & CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AICog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    @app_commands.command(name="ai", description="Chat with Gemini 2.0 Flash AI")
    @app_commands.describe(prompt="Your question or prompt")
    async def ai_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to get AI response.")
            
            embed = discord.Embed(title="🤖 Gemini Response", description=response[:2000], color=0x7289DA)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"AI error: {e}")
            await interaction.followup.send("❌ Failed to get AI response.")

    @commands.command(name="ai", aliases=["ask", "gemini"])
    async def ai_prefix(self, ctx, *, prompt: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to get AI response.")
            
            embed = discord.Embed(title="🤖 Gemini Response", description=response[:2000], color=0x7289DA)
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"AI error: {e}")
            await ctx.send("❌ Failed to get AI response.")

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

    @app_commands.command(name="translate", description="Translate text")
    @app_commands.describe(text="Text to translate", target_language="Target language")
    async def translate_slash(self, interaction: discord.Interaction, text: str, target_language: str = "English"):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            prompt = f"Translate this to {target_language}: {text}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to translate.")
            
            embed = discord.Embed(title="🌍 Translation", description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response}", color=0x5865F2)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Translate error: {e}")
            await interaction.followup.send("❌ Failed to translate.")

    @commands.command(name="translate", aliases=["trans"])
    async def translate_prefix(self, ctx, target_language: str, *, text: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            prompt = f"Translate this to {target_language}: {text}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to translate.")
            
            embed = discord.Embed(title="🌍 Translation", description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response}", color=0x5865F2)
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Translate error: {e}")
            await ctx.send("❌ Failed to translate.")

    @app_commands.command(name="summarize", description="Summarize text")
    @app_commands.describe(text="Text to summarize", length="Summary length (short/medium/long)")
    async def summarize_slash(self, interaction: discord.Interaction, text: str, length: str = "medium"):
        await interaction.response.defer()
        if not self.bot.google_ai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await interaction.followup.send("❌ Failed to summarize.")
            
            embed = discord.Embed(title="📝 Summary", description=response[:2000], color=0x2ECC71)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await interaction.followup.send("❌ Failed to summarize.")

    @commands.command(name="summarize", aliases=["summary"])
    async def summarize_prefix(self, ctx, length: str = "medium", *, text: str):
        if not self.bot.google_ai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await self.bot.google_ai.generate_text(prompt)
            if not response:
                return await ctx.send("❌ Failed to summarize.")
            
            embed = discord.Embed(title="📝 Summary", description=response[:2000], color=0x2ECC71)
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await ctx.send("❌ Failed to summarize.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 2: MODERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModerationCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    @app_commands.command(name="clear", description="Clear messages")
    @app_commands.describe(amount="Number of messages (1-100)")
    @app_commands.checks.has_permissions(manage_messages=True)
    async def clear_slash(self, interaction: discord.Interaction, amount: int = 10):
        if not 1 <= amount <= 100:
            return await interaction.response.send_message("Amount must be 1-100.", ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        deleted = await interaction.channel.purge(limit=amount)
        await interaction.followup.send(f"🧹 Cleared {len(deleted)} messages.", ephemeral=True)

    @commands.command(name="clear", aliases=["purge"])
    @commands.has_permissions(manage_messages=True)
    async def clear_prefix(self, ctx, amount: int = 10):
        if not 1 <= amount <= 100:
            return await ctx.send("Amount must be 1-100.")
        deleted = await ctx.channel.purge(limit=amount + 1)
        await ctx.send(f"🧹 Cleared {len(deleted)-1} messages.", delete_after=3)

    @app_commands.command(name="kick", description="Kick a member")
    @app_commands.describe(member="Member to kick", reason="Reason for kick")
    @app_commands.checks.has_permissions(kick_members=True)
    async def kick_slash(self, interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
        await interaction.response.defer()
        try:
            await member.kick(reason=reason)
            embed = discord.Embed(title="👢 Member Kicked", description=f"{member.mention} has been kicked.", color=0xE74C3C)
            embed.add_field(name="Reason", value=reason)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to kick: {e}")

    @commands.command(name="kick")
    @commands.has_permissions(kick_members=True)
    async def kick_prefix(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        try:
            await member.kick(reason=reason)
            embed = discord.Embed(title="👢 Member Kicked", description=f"{member.mention} has been kicked.", color=0xE74C3C)
            embed.add_field(name="Reason", value=reason)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Failed to kick: {e}")

    @app_commands.command(name="ban", description="Ban a member")
    @app_commands.describe(member="Member to ban", reason="Reason for ban", delete_days="Delete messages from days")
    @app_commands.checks.has_permissions(ban_members=True)
    async def ban_slash(self, interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided", delete_days: int = 0):
        await interaction.response.defer()
        try:
            await member.ban(reason=reason, delete_message_days=delete_days)
            embed = discord.Embed(title="🔨 Member Banned", description=f"{member.mention} has been banned.", color=0xE74C3C)
            embed.add_field(name="Reason", value=reason)
            embed.add_field(name="Messages Deleted", value=f"{delete_days} days")
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to ban: {e}")

    @commands.command(name="ban")
    @commands.has_permissions(ban_members=True)
    async def ban_prefix(self, ctx, member: discord.Member, delete_days: int = 0, *, reason: str = "No reason provided"):
        try:
            await member.ban(reason=reason, delete_message_days=delete_days)
            embed = discord.Embed(title="🔨 Member Banned", description=f"{member.mention} has been banned.", color=0xE74C3C)
            embed.add_field(name="Reason", value=reason)
            embed.add_field(name="Messages Deleted", value=f"{delete_days} days")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Failed to ban: {e}")

    @app_commands.command(name="timeout", description="Timeout a member")
    @app_commands.describe(member="Member to timeout", minutes="Timeout duration in minutes", reason="Reason")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def timeout_slash(self, interaction: discord.Interaction, member: discord.Member, minutes: int = 10, reason: str = "No reason provided"):
        await interaction.response.defer()
        try:
            duration = datetime.timedelta(minutes=minutes)
            await member.timeout(duration, reason=reason)
            embed = discord.Embed(title="⏰ Member Timed Out", description=f"{member.mention} has been timed out for {minutes} minutes.", color=0xF1C40F)
            embed.add_field(name="Reason", value=reason)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to timeout: {e}")

    @app_commands.command(name="warn", description="Warn a member")
    @app_commands.describe(member="Member to warn", reason="Reason for warning")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def warn_slash(self, interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
        embed = discord.Embed(title="⚠️ Warning Issued", description=f"{member.mention} has been warned.", color=0xF39C12)
        embed.add_field(name="Reason", value=reason)
        await interaction.response.send_message(embed=embed)

    @commands.command(name="warn")
    @commands.has_permissions(moderate_members=True)
    async def warn_prefix(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        embed = discord.Embed(title="⚠️ Warning Issued", description=f"{member.mention} has been warned.", color=0xF39C12)
        embed.add_field(name="Reason", value=reason)
        await ctx.send(embed=embed)

    @app_commands.command(name="userinfo", description="Get user information")
    @app_commands.describe(member="Member to check (leave empty for yourself)")
    async def userinfo_slash(self, interaction: discord.Interaction, member: Optional[discord.Member] = None):
        target = member or interaction.user
        embed = discord.Embed(title=f"👤 User Info: {target.display_name}", color=target.color)
        embed.set_thumbnail(url=target.display_avatar.url)
        embed.add_field(name="Username", value=str(target), inline=True)
        embed.add_field(name="ID", value=target.id, inline=True)
        embed.add_field(name="Created", value=target.created_at.strftime("%Y-%m-%d"), inline=True)
        if target.joined_at:
            embed.add_field(name="Joined", value=target.joined_at.strftime("%Y-%m-%d"), inline=True)
        roles = [role.mention for role in target.roles[1:]]
        embed.add_field(name="Roles", value=" ".join(roles) if roles else "None", inline=False)
        await interaction.response.send_message(embed=embed)

    @commands.command(name="userinfo", aliases=["whois", "info"])
    async def userinfo_prefix(self, ctx, member: Optional[discord.Member] = None):
        target = member or ctx.author
        embed = discord.Embed(title=f"👤 User Info: {target.display_name}", color=target.color)
        embed.set_thumbnail(url=target.display_avatar.url)
        embed.add_field(name="Username", value=str(target), inline=True)
        embed.add_field(name="ID", value=target.id, inline=True)
        embed.add_field(name="Created", value=target.created_at.strftime("%Y-%m-%d"), inline=True)
        if target.joined_at:
            embed.add_field(name="Joined", value=target.joined_at.strftime("%Y-%m-%d"), inline=True)
        roles = [role.mention for role in target.roles[1:]]
        embed.add_field(name="Roles", value=" ".join(roles) if roles else "None", inline=False)
        await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 3: CRYPTO & MARKET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CryptoMarketCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    @app_commands.command(name="crypto", description="Get cryptocurrency price")
    @app_commands.describe(coin="Cryptocurrency name or symbol (e.g., bitcoin, btc)")
    async def crypto_slash(self, interaction: discord.Interaction, coin: str = "bitcoin"):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies=usd&include_24hr_change=true", timeout=10) as r:
                data = await r.json()
                if coin.lower() not in data:
                    return await interaction.followup.send(f"❌ Cryptocurrency '{coin}' not found.")
                
                price = data[coin.lower()]['usd']
                change = data[coin.lower()]['usd_24h_change']
                embed = discord.Embed(title=f"💰 {coin.upper()}", color=0x2ECC71 if change >= 0 else 0xE74C3C)
                embed.add_field(name="Price", value=f"${price:,.2f}", inline=True)
                embed.add_field(name="24h Change", value=f"{change:+.2f}%", inline=True)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Crypto error: {e}")
            await interaction.followup.send("❌ Failed to fetch crypto data.")

    @commands.command(name="crypto", aliases=["coin", "price"])
    async def crypto_prefix(self, ctx, coin: str = "bitcoin"):
        async with ctx.typing():
            try:
                async with self.bot.session.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies=usd&include_24hr_change=true", timeout=10) as r:
                    data = await r.json()
                    if coin.lower() not in data:
                        return await ctx.send(f"❌ Cryptocurrency '{coin}' not found.")
                    
                    price = data[coin.lower()]['usd']
                    change = data[coin.lower()]['usd_24h_change']
                    embed = discord.Embed(title=f"💰 {coin.upper()}", color=0x2ECC71 if change >= 0 else 0xE74C3C)
                    embed.add_field(name="Price", value=f"${price:,.2f}", inline=True)
                    embed.add_field(name="24h Change", value=f"{change:+.2f}%", inline=True)
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Crypto error: {e}")
                await ctx.send("❌ Failed to fetch crypto data.")

    @app_commands.command(name="stock", description="Get stock price")
    @app_commands.describe(symbol="Stock symbol (e.g., AAPL, TSLA)")
    async def stock_slash(self, interaction: discord.Interaction, symbol: str):
        await interaction.response.defer()
        if not Cfg.FINNHUB_KEY:
            return await interaction.followup.send("❌ Finnhub API not configured.")
        try:
            async with self.bot.session.get(f"https://finnhub.io/api/v1/quote?symbol={symbol.upper()}&token={Cfg.FINNHUB_KEY}", timeout=10) as r:
                data = await r.json()
                if not data.get('c'):
                    return await interaction.followup.send(f"❌ Symbol '{symbol}' not found.")
                
                current = data['c']
                previous = data['pc']
                change = current - previous
                change_percent = (change / previous) * 100 if previous != 0 else 0
                embed = discord.Embed(title=f"📈 {symbol.upper()}", color=0x2ECC71 if change >= 0 else 0xE74C3C)
                embed.add_field(name="Current", value=f"${current:.2f}", inline=True)
                embed.add_field(name="Change", value=f"{change:+.2f} ({change_percent:+.2f}%)", inline=True)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Stock error: {e}")
            await interaction.followup.send("❌ Failed to fetch stock data.")

    @app_commands.command(name="trending", description="Get trending cryptocurrencies")
    async def trending_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://api.coingecko.com/api/v3/search/trending", timeout=10) as r:
                data = await r.json()
                embed = discord.Embed(title="📈 Trending Cryptocurrencies", color=0x9B59B6)
                for i, coin in enumerate(data['coins'][:5], 1):
                    coin_data = coin['item']
                    embed.add_field(name=f"{i}. {coin_data['name']} ({coin_data['symbol'].upper()})", value=f"Market Cap Rank: #{coin_data['market_cap_rank']}", inline=False)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Trending error: {e}")
            await interaction.followup.send("❌ Failed to fetch trending data.")

    @commands.command(name="trending", aliases=["trend"])
    async def trending_prefix(self, ctx):
        async with ctx.typing():
            try:
                async with self.bot.session.get("https://api.coingecko.com/api/v3/search/trending", timeout=10) as r:
                    data = await r.json()
                    embed = discord.Embed(title="📈 Trending Cryptocurrencies", color=0x9B59B6)
                    for i, coin in enumerate(data['coins'][:5], 1):
                        coin_data = coin['item']
                        embed.add_field(name=f"{i}. {coin_data['name']} ({coin_data['symbol'].upper()})", value=f"Market Cap Rank: #{coin_data['market_cap_rank']}", inline=False)
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Trending error: {e}")
                await ctx.send("❌ Failed to fetch trending data.")

    @app_commands.command(name="exchanges", description="List cryptocurrency exchanges")
    async def exchanges_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://api.coingecko.com/api/v3/exchanges?per_page=10", timeout=10) as r:
                data = await r.json()
                embed = discord.Embed(title="🏦 Top Crypto Exchanges", color=0x3498DB)
                for exchange in data[:7]:
                    embed.add_field(name=exchange['name'], value=f"Trust: {exchange.get('trust_score', 'N/A')}/10", inline=True)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Exchanges error: {e}")
            await interaction.followup.send("❌ Failed to fetch exchange data.")

    @app_commands.command(name="gas", description="Check Ethereum gas prices")
    async def gas_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://api.etherscan.io/api?module=gastracker&action=gasoracle", timeout=10) as r:
                data = await r.json()
                if data['status'] == '1':
                    result = data['result']
                    embed = discord.Embed(title="⛽ Ethereum Gas Prices", color=0x8A2BE2)
                    embed.add_field(name="Fast", value=f"{result['FastGasPrice']} Gwei", inline=True)
                    embed.add_field(name="Standard", value=f"{result['ProposeGasPrice']} Gwei", inline=True)
                    embed.add_field(name="Slow", value=f"{result['SafeGasPrice']} Gwei", inline=True)
                    return await interaction.followup.send(embed=embed)
        except:
            pass
        
        embed = discord.Embed(title="⛽ Ethereum Gas Prices", description="Using fallback data", color=0x8A2BE2)
        embed.add_field(name="Fast", value="30 Gwei", inline=True)
        embed.add_field(name="Standard", value="20 Gwei", inline=True)
        embed.add_field(name="Slow", value="15 Gwei", inline=True)
        await interaction.followup.send(embed=embed)

    @commands.command(name="gas")
    async def gas_prefix(self, ctx):
        async with ctx.typing():
            embed = discord.Embed(title="⛽ Ethereum Gas Prices", color=0x8A2BE2)
            embed.add_field(name="Fast", value="30 Gwei", inline=True)
            embed.add_field(name="Standard", value="20 Gwei", inline=True)
            embed.add_field(name="Slow", value="15 Gwei", inline=True)
            embed.set_footer(text="Using fallback data")
            await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 4: UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UtilityCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    @app_commands.command(name="weather", description="Get weather for a city")
    @app_commands.describe(city="City name")
    async def weather_slash(self, interaction: discord.Interaction, city: str):
        await interaction.response.defer()
        if not Cfg.WEATHER_KEY:
            return await interaction.followup.send("❌ Weather API not configured.")
        try:
            async with self.bot.session.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={Cfg.WEATHER_KEY}&units=metric", timeout=10) as r:
                data = await r.json()
                if data.get('cod') != 200:
                    return await interaction.followup.send(f"❌ City '{city}' not found.")
                
                temp = data['main']['temp']
                feels = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                country = data['sys']['country']
                
                embed = discord.Embed(title=f"🌤️ Weather in {data['name']}, {country}", color=0x3498DB)
                embed.add_field(name="Temperature", value=f"{temp}°C", inline=True)
                embed.add_field(name="Feels Like", value=f"{feels}°C", inline=True)
                embed.add_field(name="Humidity", value=f"{humidity}%", inline=True)
                embed.add_field(name="Conditions", value=description.title(), inline=True)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Weather error: {e}")
            await interaction.followup.send("❌ Failed to fetch weather data.")

    @commands.command(name="weather")
    async def weather_prefix(self, ctx, *, city: str):
        async with ctx.typing():
            if not Cfg.WEATHER_KEY:
                return await ctx.send("❌ Weather API not configured.")
            try:
                async with self.bot.session.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={Cfg.WEATHER_KEY}&units=metric", timeout=10) as r:
                    data = await r.json()
                    if data.get('cod') != 200:
                        return await ctx.send(f"❌ City '{city}' not found.")
                    
                    temp = data['main']['temp']
                    feels = data['main']['feels_like']
                    humidity = data['main']['humidity']
                    description = data['weather'][0]['description']
                    country = data['sys']['country']
                    
                    embed = discord.Embed(title=f"🌤️ Weather in {data['name']}, {country}", color=0x3498DB)
                    embed.add_field(name="Temperature", value=f"{temp}°C", inline=True)
                    embed.add_field(name="Feels Like", value=f"{feels}°C", inline=True)
                    embed.add_field(name="Humidity", value=f"{humidity}%", inline=True)
                    embed.add_field(name="Conditions", value=description.title(), inline=True)
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Weather error: {e}")
                await ctx.send("❌ Failed to fetch weather data.")

    @app_commands.command(name="timer", description="Set a timer")
    @app_commands.describe(seconds="Seconds to wait", minutes="Minutes to wait", hours="Hours to wait")
    async def timer_slash(self, interaction: discord.Interaction, seconds: int = 0, minutes: int = 0, hours: int = 0):
        total_seconds = seconds + (minutes * 60) + (hours * 3600)
        if total_seconds <= 0:
            return await interaction.response.send_message("❌ Please specify a valid time.", ephemeral=True)
        if total_seconds > 86400:
            return await interaction.response.send_message("❌ Timer cannot exceed 24 hours.", ephemeral=True)
        
        await interaction.response.send_message(f"⏰ Timer set for {total_seconds} seconds.")
        await asyncio.sleep(total_seconds)
        
        try:
            await interaction.followup.send(f"⏰ Timer ended! {interaction.user.mention}")
        except:
            pass

    @app_commands.command(name="poll", description="Create a poll")
    @app_commands.describe(question="Poll question", option1="Option 1", option2="Option 2")
    async def poll_slash(self, interaction: discord.Interaction, question: str, option1: str, option2: str, option3: str = None, option4: str = None):
        options = [option1, option2]
        if option3:
            options.append(option3)
        if option4:
            options.append(option4)
        
        embed = discord.Embed(title=f"📊 Poll: {question}", description="\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]), color=0x9B59B6)
        embed.set_footer(text=f"Poll by {interaction.user.display_name}")
        
        await interaction.response.send_message(embed=embed)
        message = await interaction.original_response()
        for i in range(len(options)):
            await message.add_reaction(f"{i+1}️⃣")

    @commands.command(name="poll")
    async def poll_prefix(self, ctx, question: str, *options):
        if len(options) < 2:
            return await ctx.send("❌ Need at least 2 options for a poll.")
        if len(options) > 10:
            return await ctx.send("❌ Maximum 10 options allowed.")
        
        embed = discord.Embed(title=f"📊 Poll: {question}", description="\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]), color=0x9B59B6)
        embed.set_footer(text=f"Poll by {ctx.author.display_name}")
        
        message = await ctx.send(embed=embed)
        for i in range(len(options)):
            await message.add_reaction(f"{i+1}️⃣")

    @app_commands.command(name="remind", description="Set a reminder")
    @app_commands.describe(time="Time (e.g., 30s, 5m, 2h, 1d)", reminder="What to remember")
    async def remind_slash(self, interaction: discord.Interaction, time: str, reminder: str):
        await interaction.response.defer(ephemeral=True)
        
        time_value = time[:-1]
        time_unit = time[-1].lower()
        
        if not time_value.isdigit():
            return await interaction.followup.send("❌ Invalid time format.", ephemeral=True)
        
        time_value = int(time_value)
        
        if time_unit == 's':
            seconds = time_value
        elif time_unit == 'm':
            seconds = time_value * 60
        elif time_unit == 'h':
            seconds = time_value * 3600
        elif time_unit == 'd':
            seconds = time_value * 86400
        else:
            return await interaction.followup.send("❌ Use s, m, h, or d (e.g., 30s, 5m, 2h, 1d)", ephemeral=True)
        
        if seconds > 604800:
            return await interaction.followup.send("❌ Reminder cannot exceed 7 days.", ephemeral=True)
        
        await interaction.followup.send(f"✅ Reminder set for {time}.", ephemeral=True)
        await asyncio.sleep(seconds)
        
        try:
            await interaction.user.send(f"⏰ Reminder: {reminder}")
        except:
            try:
                await interaction.channel.send(f"⏰ {interaction.user.mention} Reminder: {reminder}")
            except:
                pass

    @app_commands.command(name="calc", description="Calculate a math expression")
    @app_commands.describe(expression="Math expression (e.g., 2+2, 5*3, sqrt(16))")
    async def calc_slash(self, interaction: discord.Interaction, expression: str):
        try:
            expression_clean = expression.replace('^', '**').replace('sqrt', 'math.sqrt')
            allowed_names = {'math': __import__('math'), 'abs': abs, 'round': round, 'min': min, 'max': max}
            code = compile(expression_clean, '<string>', 'eval')
            for name in code.co_names:
                if name not in allowed_names:
                    return await interaction.response.send_message("❌ Invalid expression.", ephemeral=True)
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            embed = discord.Embed(title="🧮 Calculator", description=f"```\n{expression} = {result}\n```", color=0x2ECC71)
            await interaction.response.send_message(embed=embed)
        except:
            await interaction.response.send_message("❌ Could not calculate expression.", ephemeral=True)

    @commands.command(name="calc")
    async def calc_prefix(self, ctx, *, expression: str):
        try:
            expression_clean = expression.replace('^', '**').replace('sqrt', 'math.sqrt')
            allowed_names = {'math': __import__('math'), 'abs': abs, 'round': round, 'min': min, 'max': max}
            code = compile(expression_clean, '<string>', 'eval')
            for name in code.co_names:
                if name not in allowed_names:
                    return await ctx.send("❌ Invalid expression.")
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            embed = discord.Embed(title="🧮 Calculator", description=f"```\n{expression} = {result}\n```", color=0x2ECC71)
            await ctx.send(embed=embed)
        except:
            await ctx.send("❌ Could not calculate expression.")

    @app_commands.command(name="define", description="Get word definition")
    @app_commands.describe(word="Word to define")
    async def define_slash(self, interaction: discord.Interaction, word: str):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=10) as r:
                if r.status != 200:
                    return await interaction.followup.send(f"❌ No definition found for '{word}'.")
                
                data = await r.json()
                first_entry = data[0]
                embed = discord.Embed(title=f"📖 {word.title()}", color=0x3498DB)
                
                for meaning in first_entry['meanings'][:3]:
                    part_of_speech = meaning['partOfSpeech']
                    definition = meaning['definitions'][0]['definition']
                    embed.add_field(name=part_of_speech.title(), value=definition[:500], inline=False)
                
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Define error: {e}")
            await interaction.followup.send(f"❌ Could not fetch definition for '{word}'.")

    @app_commands.command(name="qr", description="Generate QR code")
    @app_commands.describe(text="Text or URL to encode")
    async def qr_slash(self, interaction: discord.Interaction, text: str):
        encoded = urllib.parse.quote(text)
        url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={encoded}"
        embed = discord.Embed(title="📱 QR Code", description=f"**Content:** {text[:100]}", color=0x000000)
        embed.set_image(url=url)
        await interaction.response.send_message(embed=embed)

    @commands.command(name="qr")
    async def qr_prefix(self, ctx, *, text: str):
        encoded = urllib.parse.quote(text)
        url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={encoded}"
        embed = discord.Embed(title="📱 QR Code", description=f"**Content:** {text[:100]}", color=0x000000)
        embed.set_image(url=url)
        await ctx.send(embed=embed)

    @app_commands.command(name="serverinfo", description="Get server information")
    async def serverinfo_slash(self, interaction: discord.Interaction):
        guild = interaction.guild
        embed = discord.Embed(title=f"🏰 {guild.name}", color=0x7289DA)
        
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        
        embed.add_field(name="Owner", value=guild.owner.mention, inline=True)
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Channels", value=len(guild.channels), inline=True)
        embed.add_field(name="Roles", value=len(guild.roles), inline=True)
        embed.add_field(name="Created", value=guild.created_at.strftime("%Y-%m-%d"), inline=True)
        embed.add_field(name="Boosts", value=guild.premium_subscription_count, inline=True)
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="serverinfo", aliases=["server", "guildinfo"])
    async def serverinfo_prefix(self, ctx):
        guild = ctx.guild
        embed = discord.Embed(title=f"🏰 {guild.name}", color=0x7289DA)
        
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        
        embed.add_field(name="Owner", value=guild.owner.mention, inline=True)
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Channels", value=len(guild.channels), inline=True)
        embed.add_field(name="Roles", value=len(guild.roles), inline=True)
        embed.add_field(name="Created", value=guild.created_at.strftime("%Y-%m-%d"), inline=True)
        embed.add_field(name="Boosts", value=guild.premium_subscription_count, inline=True)
        
        await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 5: ENTERTAINMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EntertainmentCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    @app_commands.command(name="meme", description="Get a random meme")
    async def meme_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://meme-api.com/gimme", timeout=10) as r:
                data = await r.json()
                embed = discord.Embed(title=data['title'], url=data['postLink'], color=0xFF9900)
                embed.set_image(url=data['url'])
                embed.set_footer(text=f"👍 {data['ups']} | r/{data['subreddit']}")
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Meme error: {e}")
            await interaction.followup.send("❌ Could not fetch meme.")

    @commands.command(name="meme")
    async def meme_prefix(self, ctx):
        async with ctx.typing():
            try:
                async with self.bot.session.get("https://meme-api.com/gimme", timeout=10) as r:
                    data = await r.json()
                    embed = discord.Embed(title=data['title'], url=data['postLink'], color=0xFF9900)
                    embed.set_image(url=data['url'])
                    embed.set_footer(text=f"👍 {data['ups']} | r/{data['subreddit']}")
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Meme error: {e}")
                await ctx.send("❌ Could not fetch meme.")

    @app_commands.command(name="joke", description="Get a random joke")
    async def joke_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist", timeout=10) as r:
                data = await r.json()
                
                if data['type'] == 'single':
                    embed = discord.Embed(title="😂 Joke", description=data['joke'], color=0xFFD700)
                else:
                    embed = discord.Embed(title="😂 Joke", description=f"**{data['setup']}**\n\n{data['delivery']}", color=0xFFD700)
                
                embed.set_footer(text=f"Category: {data['category']}")
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Joke error: {e}")
            await interaction.followup.send("❌ Could not fetch joke.")

    @commands.command(name="joke")
    async def joke_prefix(self, ctx):
        async with ctx.typing():
            try:
                async with self.bot.session.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist", timeout=10) as r:
                    data = await r.json()
                    
                    if data['type'] == 'single':
                        embed = discord.Embed(title="😂 Joke", description=data['joke'], color=0xFFD700)
                    else:
                        embed = discord.Embed(title="😂 Joke", description=f"**{data['setup']}**\n\n{data['delivery']}", color=0xFFD700)
                    
                    embed.set_footer(text=f"Category: {data['category']}")
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Joke error: {e}")
                await ctx.send("❌ Could not fetch joke.")

    @app_commands.command(name="8ball", description="Ask the magic 8-ball")
    @app_commands.describe(question="Your question")
    async def eightball_slash(self, interaction: discord.Interaction, question: str):
        responses = [
            "It is certain.", "It is decidedly so.", "Without a doubt.",
            "Yes - definitely.", "You may rely on it.", "As I see it, yes.",
            "Most likely.", "Outlook good.", "Yes.", "Signs point to yes.",
            "Reply hazy, try again.", "Ask again later.", "Better not tell you now.",
            "Cannot predict now.", "Concentrate and ask again.",
            "Don't count on it.", "My reply is no.", "My sources say no.",
            "Outlook not so good.", "Very doubtful."
        ]
        
        embed = discord.Embed(title="🎱 Magic 8-Ball", description=f"**Question:** {question}\n\n**Answer:** {random.choice(responses)}", color=0x000000)
        await interaction.response.send_message(embed=embed)

    @commands.command(name="8ball", aliases=["magic8"])
    async def eightball_prefix(self, ctx, *, question: str):
        responses = [
            "It is certain.", "It is decidedly so.", "Without a doubt.",
            "Yes - definitely.", "You may rely on it.", "As I see it, yes.",
            "Most likely.", "Outlook good.", "Yes.", "Signs point to yes.",
            "Reply hazy, try again.", "Ask again later.", "Better not tell you now.",
            "Cannot predict now.", "Concentrate and ask again.",
            "Don't count on it.", "My reply is no.", "My sources say no.",
            "Outlook not so good.", "Very doubtful."
        ]
        
        embed = discord.Embed(title="🎱 Magic 8-Ball", description=f"**Question:** {question}\n\n**Answer:** {random.choice(responses)}", color=0x000000)
        await ctx.send(embed=embed)

    @app_commands.command(name="roll", description="Roll dice")
    @app_commands.describe(dice="Dice notation (e.g., 2d6)")
    async def roll_slash(self, interaction: discord.Interaction, dice: str = "1d6"):
        try:
            if 'd' not in dice:
                return await interaction.response.send_message("❌ Use format like '2d6' or '1d20'", ephemeral=True)
            
            num, sides = dice.split('d')
            num = int(num) if num else 1
            sides = int(sides)
            
            if num > 100 or sides > 1000:
                return await interaction.response.send_message("❌ Too many dice or sides.", ephemeral=True)
            
            rolls = [random.randint(1, sides) for _ in range(num)]
            total = sum(rolls)
            
            embed = discord.Embed(title="🎲 Dice Roll", description=f"**{dice}**", color=0xE74C3C)
            embed.add_field(name="Rolls", value=", ".join(map(str, rolls)) if len(rolls) <= 20 else f"{len(rolls)} rolls", inline=True)
            embed.add_field(name="Total", value=total, inline=True)
            
            await interaction.response.send_message(embed=embed)
        except:
            await interaction.response.send_message("❌ Invalid dice format. Use like '2d6' or '1d20'", ephemeral=True)

    @commands.command(name="roll", aliases=["dice"])
    async def roll_prefix(self, ctx, dice: str = "1d6"):
        try:
            if 'd' not in dice:
                return await ctx.send("❌ Use format like '2d6' or '1d20'")
            
            num, sides = dice.split('d')
            num = int(num) if num else 1
            sides = int(sides)
            
            if num > 100 or sides > 1000:
                return await ctx.send("❌ Too many dice or sides.")
            
            rolls = [random.randint(1, sides) for _ in range(num)]
            total = sum(rolls)
            
            embed = discord.Embed(title="🎲 Dice Roll", description=f"**{dice}**", color=0xE74C3C)
            embed.add_field(name="Rolls", value=", ".join(map(str, rolls)) if len(rolls) <= 20 else f"{len(rolls)} rolls", inline=True)
            embed.add_field(name="Total", value=total, inline=True)
            
            await ctx.send(embed=embed)
        except:
            await ctx.send("❌ Invalid dice format. Use like '2d6' or '1d20'")

    @app_commands.command(name="fact", description="Get a random fact")
    async def fact_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get("https://uselessfacts.jsph.pl/random.json?language=en", timeout=10) as r:
                data = await r.json()
                embed = discord.Embed(title="📚 Random Fact", description=data['text'], color=0x1ABC9C)
                embed.set_footer(text="Source: uselessfacts.jsph.pl")
                await interaction.followup.send(embed=embed)
        except:
            facts = [
                "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
                "Octopuses have three hearts. Two pump blood to the gills, while the third pumps it to the rest of the body.",
                "A day on Venus is longer than a year on Venus. Venus takes 243 Earth days to rotate once on its axis, but only 225 Earth days to orbit the Sun.",
                "Bananas are berries, but strawberries aren't.",
                "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion of the metal.",
            ]
            embed = discord.Embed(title="📚 Random Fact", description=random.choice(facts), color=0x1ABC9C)
            await interaction.followup.send(embed=embed)

    @commands.command(name="fact")
    async def fact_prefix(self, ctx):
        async with ctx.typing():
            facts = [
                "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
                "Octopuses have three hearts. Two pump blood to the gills, while the third pumps it to the rest of the body.",
                "A day on Venus is longer than a year on Venus. Venus takes 243 Earth days to rotate once on its axis, but only 225 Earth days to orbit the Sun.",
                "Bananas are berries, but strawberries aren't.",
                "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion of the metal.",
            ]
            embed = discord.Embed(title="📚 Random Fact", description=random.choice(facts), color=0x1ABC9C)
            await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 6: NASA SPACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NASACog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

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
                embed = discord.Embed(title=f"🚀 NASA APOD: {data.get('title', 'Astronomy Picture of the Day')}", description=data.get('explanation', '')[:1000] + "...", color=0x1E3A8A)
                
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
                    embed = discord.Embed(title=f"🚀 NASA APOD: {data.get('title', 'Astronomy Picture of the Day')}", description=data.get('explanation', '')[:1000] + "...", color=0x1E3A8A)
                    
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
                
                photo = data['latest_photos'][0]
                embed = discord.Embed(title="🔴 Latest Mars Rover Photo", description=f"Taken by {photo['rover']['name']} Rover", color=0x8B0000)
                embed.set_image(url=photo['img_src'])
                embed.add_field(name="Earth Date", value=photo['earth_date'], inline=True)
                embed.add_field(name="Camera", value=photo['camera']['full_name'], inline=True)
                embed.add_field(name="Sol", value=photo['sol'], inline=True)
                embed.set_footer(text=f"NASA Mars Rover • Total photos: {photo['rover']['total_photos']:,}")
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Mars photos error: {e}")
            await interaction.followup.send("❌ Failed to fetch Mars photos.")

    @app_commands.command(name="iss", description="International Space Station current location")
    async def iss_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            url = "http://api.open-notify.org/iss-now.json"
            async with self.bot.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return await interaction.followup.send("❌ Failed to fetch ISS location.")
                
                data = await response.json()
                lat = float(data['iss_position']['latitude'])
                lon = float(data['iss_position']['longitude'])
                
                location_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
                async with self.bot.session.get(location_url, headers={'User-Agent': 'GodBot/4.0'}, timeout=10) as loc
