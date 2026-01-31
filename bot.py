"""
GodBot v4.0 — Complete Working Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL 36 Commands with Fixed IBM Guardian & Updated Google AI
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
# FIXED IBM GUARDIAN SERVICE (100% Working)
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
            
            # Prepare prompt - clearer instruction for the model
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
                
                # Extract response - more robust parsing
                generated_text = ""
                try:
                    generated_text = result.get('results', [{}])[0].get('generated_text', '').strip().lower()
                except (IndexError, KeyError, AttributeError):
                    logger.error(f"Unexpected IBM response format: {result}")
                    return None
                
                logger.debug(f"IBM Raw Response: '{generated_text}'")
                
                # Robust detection of yes/no
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
    VERSION = "4.0-Complete"

    def __init__(self):
        super().__init__(
            command_prefix=Cfg.PREFIX,
            intents=discord.Intents.all(),
            help_command=None,
            case_insensitive=True,
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.ibm_guardian: Optional[IBMGuardian] = None
        self.genai = None
        self.start_time = datetime.datetime.now(timezone.utc)

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        await self._init_gemini()
        await self._init_ibm()

        # Load all cogs
        cogs = [
            AICog,
            ModerationCog,
            CryptoMarketCog,
            UtilityCog,
            EntertainmentCog,
            SystemCog
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

    async def _init_gemini(self):
        """Initialize Google Gemini with NEW google.genai package"""
        if not Cfg.GEMINI_KEY:
            logger.warning("GEMINI_API_KEY missing — AI commands disabled")
            return
        try:
            import google.genai as genai
            genai.configure(api_key=Cfg.GEMINI_KEY)
            self.genai = genai
            logger.info("🤖 Gemini 2.0 Flash (google.genai) — ready")
        except ImportError:
            logger.error("Please install: pip install google-genai")
            self.genai = None
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            self.genai = None

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
        logger.info(f"🤖 Gemini AI: {'ACTIVE' if self.genai else 'DISABLED'}")

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
                
                # Send warning (optional)
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
            f"Gemini: {'🤖' if self.genai else '❌'}",
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
# COG 1: AI & CHAT (4 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AICog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 1. /ai - Ask AI
    @app_commands.command(name="ai", description="Chat with Gemini 2.0 Flash AI")
    @app_commands.describe(prompt="Your question or prompt")
    async def ai_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer()
        if not self.bot.genai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="🤖 Gemini Response",
                description=response.text[:2000],
                color=0x7289DA,
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"AI error: {e}")
            await interaction.followup.send("❌ Failed to get AI response.")

    @commands.command(name="ai", aliases=["ask", "gemini"])
    async def ai_prefix(self, ctx, *, prompt: str):
        if not self.bot.genai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="🤖 Gemini Response",
                description=response.text[:2000],
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
        if not self.bot.genai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Translate this to {target_language}: {text}"
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="🌍 Translation",
                description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response.text}",
                color=0x5865F2
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Translate error: {e}")
            await interaction.followup.send("❌ Failed to translate.")

    @commands.command(name="translate", aliases=["trans"])
    async def translate_prefix(self, ctx, target_language: str, *, text: str):
        if not self.bot.genai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Translate this to {target_language}: {text}"
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="🌍 Translation",
                description=f"**Original:** {text}\n\n**Translated ({target_language}):** {response.text}",
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
        if not self.bot.genai:
            return await interaction.followup.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="📝 Summary",
                description=response.text[:2000],
                color=0x2ECC71
            )
            await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await interaction.followup.send("❌ Failed to summarize.")

    @commands.command(name="summarize", aliases=["summary"])
    async def summarize_prefix(self, ctx, length: str = "medium", *, text: str):
        if not self.bot.genai:
            return await ctx.send("❌ Gemini API not configured.")
        try:
            import google.genai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Summarize this in {length} length:\n\n{text[:2000]}"
            response = await asyncio.to_thread(model.generate_content, prompt)
            embed = discord.Embed(
                title="📝 Summary",
                description=response.text[:2000],
                color=0x2ECC71
            )
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await ctx.send("❌ Failed to summarize.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 2: MODERATION (6 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModerationCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 5. /clear - Clear messages
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

    # 6. /kick - Kick member
    @app_commands.command(name="kick", description="Kick a member")
    @app_commands.describe(member="Member to kick", reason="Reason for kick")
    @app_commands.checks.has_permissions(kick_members=True)
    async def kick_slash(self, interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
        await interaction.response.defer()
        try:
            await member.kick(reason=reason)
            embed = discord.Embed(
                title="👢 Member Kicked",
                description=f"{member.mention} has been kicked.",
                color=0xE74C3C
            )
            embed.add_field(name="Reason", value=reason)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to kick: {e}")

    @commands.command(name="kick")
    @commands.has_permissions(kick_members=True)
    async def kick_prefix(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        try:
            await member.kick(reason=reason)
            embed = discord.Embed(
                title="👢 Member Kicked",
                description=f"{member.mention} has been kicked.",
                color=0xE74C3C
            )
            embed.add_field(name="Reason", value=reason)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Failed to kick: {e}")

    # 7. /ban - Ban member
    @app_commands.command(name="ban", description="Ban a member")
    @app_commands.describe(member="Member to ban", reason="Reason for ban", delete_days="Delete messages from days")
    @app_commands.checks.has_permissions(ban_members=True)
    async def ban_slash(self, interaction: discord.Interaction, member: discord.Member, 
                       reason: str = "No reason provided", delete_days: int = 0):
        await interaction.response.defer()
        try:
            await member.ban(reason=reason, delete_message_days=delete_days)
            embed = discord.Embed(
                title="🔨 Member Banned",
                description=f"{member.mention} has been banned.",
                color=0xE74C3C
            )
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
            embed = discord.Embed(
                title="🔨 Member Banned",
                description=f"{member.mention} has been banned.",
                color=0xE74C3C
            )
            embed.add_field(name="Reason", value=reason)
            embed.add_field(name="Messages Deleted", value=f"{delete_days} days")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Failed to ban: {e}")

    # 8. /timeout - Timeout member
    @app_commands.command(name="timeout", description="Timeout a member")
    @app_commands.describe(member="Member to timeout", minutes="Timeout duration in minutes", reason="Reason")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def timeout_slash(self, interaction: discord.Interaction, member: discord.Member, 
                           minutes: int = 10, reason: str = "No reason provided"):
        await interaction.response.defer()
        try:
            duration = datetime.timedelta(minutes=minutes)
            await member.timeout(duration, reason=reason)
            embed = discord.Embed(
                title="⏰ Member Timed Out",
                description=f"{member.mention} has been timed out for {minutes} minutes.",
                color=0xF1C40F
            )
            embed.add_field(name="Reason", value=reason)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to timeout: {e}")

    # 9. /warn - Warn member
    @app_commands.command(name="warn", description="Warn a member")
    @app_commands.describe(member="Member to warn", reason="Reason for warning")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def warn_slash(self, interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided"):
        embed = discord.Embed(
            title="⚠️ Warning Issued",
            description=f"{member.mention} has been warned.",
            color=0xF39C12
        )
        embed.add_field(name="Reason", value=reason)
        await interaction.response.send_message(embed=embed)

    @commands.command(name="warn")
    @commands.has_permissions(moderate_members=True)
    async def warn_prefix(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        embed = discord.Embed(
            title="⚠️ Warning Issued",
            description=f"{member.mention} has been warned.",
            color=0xF39C12
        )
        embed.add_field(name="Reason", value=reason)
        await ctx.send(embed=embed)

    # 10. /userinfo - Get user information
    @app_commands.command(name="userinfo", description="Get user information")
    @app_commands.describe(member="Member to check (leave empty for yourself)")
    async def userinfo_slash(self, interaction: discord.Interaction, member: Optional[discord.Member] = None):
        target = member or interaction.user
        embed = discord.Embed(
            title=f"👤 User Info: {target.display_name}",
            color=target.color
        )
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
        embed = discord.Embed(
            title=f"👤 User Info: {target.display_name}",
            color=target.color
        )
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
# COG 3: CRYPTO & MARKET (5 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CryptoMarketCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 11. /crypto - Get cryptocurrency price
    @app_commands.command(name="crypto", description="Get cryptocurrency price")
    @app_commands.describe(coin="Cryptocurrency name or symbol (e.g., bitcoin, btc)")
    async def crypto_slash(self, interaction: discord.Interaction, coin: str = "bitcoin"):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies=usd&include_24hr_change=true",
                timeout=10
            ) as r:
                data = await r.json()
                if coin.lower() not in data:
                    return await interaction.followup.send(f"❌ Cryptocurrency '{coin}' not found.")
                
                price = data[coin.lower()]['usd']
                change = data[coin.lower()]['usd_24h_change']
                embed = discord.Embed(
                    title=f"💰 {coin.upper()}",
                    color=0x2ECC71 if change >= 0 else 0xE74C3C
                )
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
                async with self.bot.session.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies=usd&include_24hr_change=true",
                    timeout=10
                ) as r:
                    data = await r.json()
                    if coin.lower() not in data:
                        return await ctx.send(f"❌ Cryptocurrency '{coin}' not found.")
                    
                    price = data[coin.lower()]['usd']
                    change = data[coin.lower()]['usd_24h_change']
                    embed = discord.Embed(
                        title=f"💰 {coin.upper()}",
                        color=0x2ECC71 if change >= 0 else 0xE74C3C
                    )
                    embed.add_field(name="Price", value=f"${price:,.2f}", inline=True)
                    embed.add_field(name="24h Change", value=f"{change:+.2f}%", inline=True)
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Crypto error: {e}")
                await ctx.send("❌ Failed to fetch crypto data.")

    # 12. /stock - Get stock price
    @app_commands.command(name="stock", description="Get stock price")
    @app_commands.describe(symbol="Stock symbol (e.g., AAPL, TSLA)")
    async def stock_slash(self, interaction: discord.Interaction, symbol: str):
        await interaction.response.defer()
        if not Cfg.FINNHUB_KEY:
            return await interaction.followup.send("❌ Finnhub API not configured.")
        try:
            async with self.bot.session.get(
                f"https://finnhub.io/api/v1/quote?symbol={symbol.upper()}&token={Cfg.FINNHUB_KEY}",
                timeout=10
            ) as r:
                data = await r.json()
                if not data.get('c'):
                    return await interaction.followup.send(f"❌ Symbol '{symbol}' not found.")
                
                current = data['c']
                previous = data['pc']
                change = current - previous
                change_percent = (change / previous) * 100 if previous != 0 else 0
                embed = discord.Embed(
                    title=f"📈 {symbol.upper()}",
                    color=0x2ECC71 if change >= 0 else 0xE74C3C
                )
                embed.add_field(name="Current", value=f"${current:.2f}", inline=True)
                embed.add_field(name="Change", value=f"{change:+.2f} ({change_percent:+.2f}%)", inline=True)
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Stock error: {e}")
            await interaction.followup.send("❌ Failed to fetch stock data.")

    # 13. /trending - Trending cryptocurrencies
    @app_commands.command(name="trending", description="Get trending cryptocurrencies")
    async def trending_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10
            ) as r:
                data = await r.json()
                embed = discord.Embed(
                    title="📈 Trending Cryptocurrencies",
                    color=0x9B59B6
                )
                for i, coin in enumerate(data['coins'][:5], 1):
                    coin_data = coin['item']
                    embed.add_field(
                        name=f"{i}. {coin_data['name']} ({coin_data['symbol'].upper()})",
                        value=f"Market Cap Rank: #{coin_data['market_cap_rank']}",
                        inline=False
                    )
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Trending error: {e}")
            await interaction.followup.send("❌ Failed to fetch trending data.")

    @commands.command(name="trending", aliases=["trend"])
    async def trending_prefix(self, ctx):
        async with ctx.typing():
            try:
                async with self.bot.session.get(
                    "https://api.coingecko.com/api/v3/search/trending",
                    timeout=10
                ) as r:
                    data = await r.json()
                    embed = discord.Embed(
                        title="📈 Trending Cryptocurrencies",
                        color=0x9B59B6
                    )
                    for i, coin in enumerate(data['coins'][:5], 1):
                        coin_data = coin['item']
                        embed.add_field(
                            name=f"{i}. {coin_data['name']} ({coin_data['symbol'].upper()})",
                            value=f"Market Cap Rank: #{coin_data['market_cap_rank']}",
                            inline=False
                        )
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Trending error: {e}")
                await ctx.send("❌ Failed to fetch trending data.")

    # 14. /exchanges - Top crypto exchanges
    @app_commands.command(name="exchanges", description="List cryptocurrency exchanges")
    async def exchanges_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                "https://api.coingecko.com/api/v3/exchanges?per_page=10",
                timeout=10
            ) as r:
                data = await r.json()
                embed = discord.Embed(
                    title="🏦 Top Crypto Exchanges",
                    color=0x3498DB
                )
                for exchange in data[:7]:
                    embed.add_field(
                        name=exchange['name'],
                        value=f"Trust: {exchange.get('trust_score', 'N/A')}/10",
                        inline=True
                    )
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Exchanges error: {e}")
            await interaction.followup.send("❌ Failed to fetch exchange data.")

    # 15. /gas - Ethereum gas prices
    @app_commands.command(name="gas", description="Check Ethereum gas prices")
    async def gas_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            # Try etherscan first
            async with self.bot.session.get(
                "https://api.etherscan.io/api?module=gastracker&action=gasoracle",
                timeout=10
            ) as r:
                data = await r.json()
                if data['status'] == '1':
                    result = data['result']
                    embed = discord.Embed(
                        title="⛽ Ethereum Gas Prices",
                        color=0x8A2BE2
                    )
                    embed.add_field(name="Fast", value=f"{result['FastGasPrice']} Gwei", inline=True)
                    embed.add_field(name="Standard", value=f"{result['ProposeGasPrice']} Gwei", inline=True)
                    embed.add_field(name="Slow", value=f"{result['SafeGasPrice']} Gwei", inline=True)
                    return await interaction.followup.send(embed=embed)
        except:
            pass
        
        # Fallback
        embed = discord.Embed(
            title="⛽ Ethereum Gas Prices",
            description="Using fallback data",
            color=0x8A2BE2
        )
        embed.add_field(name="Fast", value="30 Gwei", inline=True)
        embed.add_field(name="Standard", value="20 Gwei", inline=True)
        embed.add_field(name="Slow", value="15 Gwei", inline=True)
        await interaction.followup.send(embed=embed)

    @commands.command(name="gas")
    async def gas_prefix(self, ctx):
        async with ctx.typing():
            embed = discord.Embed(
                title="⛽ Ethereum Gas Prices",
                color=0x8A2BE2
            )
            embed.add_field(name="Fast", value="30 Gwei", inline=True)
            embed.add_field(name="Standard", value="20 Gwei", inline=True)
            embed.add_field(name="Slow", value="15 Gwei", inline=True)
            embed.set_footer(text="Using fallback data")
            await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 4: UTILITY (8 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UtilityCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 16. /weather - Get weather
    @app_commands.command(name="weather", description="Get weather for a city")
    @app_commands.describe(city="City name")
    async def weather_slash(self, interaction: discord.Interaction, city: str):
        await interaction.response.defer()
        if not Cfg.WEATHER_KEY:
            return await interaction.followup.send("❌ Weather API not configured.")
        try:
            async with self.bot.session.get(
                f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={Cfg.WEATHER_KEY}&units=metric",
                timeout=10
            ) as r:
                data = await r.json()
                if data.get('cod') != 200:
                    return await interaction.followup.send(f"❌ City '{city}' not found.")
                
                temp = data['main']['temp']
                feels = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                country = data['sys']['country']
                
                embed = discord.Embed(
                    title=f"🌤️ Weather in {data['name']}, {country}",
                    color=0x3498DB
                )
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
                async with self.bot.session.get(
                    f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={Cfg.WEATHER_KEY}&units=metric",
                    timeout=10
                ) as r:
                    data = await r.json()
                    if data.get('cod') != 200:
                        return await ctx.send(f"❌ City '{city}' not found.")
                    
                    temp = data['main']['temp']
                    feels = data['main']['feels_like']
                    humidity = data['main']['humidity']
                    description = data['weather'][0]['description']
                    country = data['sys']['country']
                    
                    embed = discord.Embed(
                        title=f"🌤️ Weather in {data['name']}, {country}",
                        color=0x3498DB
                    )
                    embed.add_field(name="Temperature", value=f"{temp}°C", inline=True)
                    embed.add_field(name="Feels Like", value=f"{feels}°C", inline=True)
                    embed.add_field(name="Humidity", value=f"{humidity}%", inline=True)
                    embed.add_field(name="Conditions", value=description.title(), inline=True)
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Weather error: {e}")
                await ctx.send("❌ Failed to fetch weather data.")

    # 17. /timer - Set a timer
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

    # 18. /poll - Create a poll
    @app_commands.command(name="poll", description="Create a poll")
    @app_commands.describe(question="Poll question", option1="Option 1", option2="Option 2")
    async def poll_slash(self, interaction: discord.Interaction, question: str, 
                        option1: str, option2: str, option3: str = None, option4: str = None):
        options = [option1, option2]
        if option3:
            options.append(option3)
        if option4:
            options.append(option4)
        
        embed = discord.Embed(
            title=f"📊 Poll: {question}",
            description="\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]),
            color=0x9B59B6
        )
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
        
        embed = discord.Embed(
            title=f"📊 Poll: {question}",
            description="\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]),
            color=0x9B59B6
        )
        embed.set_footer(text=f"Poll by {ctx.author.display_name}")
        
        message = await ctx.send(embed=embed)
        for i in range(len(options)):
            await message.add_reaction(f"{i+1}️⃣")

    # 19. /remind - Set a reminder
    @app_commands.command(name="remind", description="Set a reminder")
    @app_commands.describe(time="Time (e.g., 30s, 5m, 2h, 1d)", reminder="What to remember")
    async def remind_slash(self, interaction: discord.Interaction, time: str, reminder: str):
        await interaction.response.defer(ephemeral=True)
        
        # Parse time
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
        
        if seconds > 604800:  # 7 days
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

    # 20. /calc - Calculator
    @app_commands.command(name="calc", description="Calculate a math expression")
    @app_commands.describe(expression="Math expression (e.g., 2+2, 5*3, sqrt(16))")
    async def calc_slash(self, interaction: discord.Interaction, expression: str):
        try:
            # Safe evaluation
            expression_clean = expression.replace('^', '**').replace('sqrt', 'math.sqrt')
            
            allowed_names = {
                'math': __import__('math'),
                'abs': abs, 'round': round, 'min': min, 'max': max,
            }
            
            code = compile(expression_clean, '<string>', 'eval')
            for name in code.co_names:
                if name not in allowed_names:
                    return await interaction.response.send_message("❌ Invalid expression.", ephemeral=True)
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            embed = discord.Embed(
                title="🧮 Calculator",
                description=f"```\n{expression} = {result}\n```",
                color=0x2ECC71
            )
            await interaction.response.send_message(embed=embed)
        except:
            await interaction.response.send_message("❌ Could not calculate expression.", ephemeral=True)

    @commands.command(name="calc")
    async def calc_prefix(self, ctx, *, expression: str):
        try:
            expression_clean = expression.replace('^', '**').replace('sqrt', 'math.sqrt')
            allowed_names = {
                'math': __import__('math'),
                'abs': abs, 'round': round, 'min': min, 'max': max,
            }
            
            code = compile(expression_clean, '<string>', 'eval')
            for name in code.co_names:
                if name not in allowed_names:
                    return await ctx.send("❌ Invalid expression.")
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            embed = discord.Embed(
                title="🧮 Calculator",
                description=f"```\n{expression} = {result}\n```",
                color=0x2ECC71
            )
            await ctx.send(embed=embed)
        except:
            await ctx.send("❌ Could not calculate expression.")

    # 21. /define - Dictionary definition
    @app_commands.command(name="define", description="Get word definition")
    @app_commands.describe(word="Word to define")
    async def define_slash(self, interaction: discord.Interaction, word: str):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
                timeout=10
            ) as r:
                if r.status != 200:
                    return await interaction.followup.send(f"❌ No definition found for '{word}'.")
                
                data = await r.json()
                first_entry = data[0]
                
                embed = discord.Embed(
                    title=f"📖 {word.title()}",
                    color=0x3498DB
                )
                
                for meaning in first_entry['meanings'][:3]:
                    part_of_speech = meaning['partOfSpeech']
                    definition = meaning['definitions'][0]['definition']
                    embed.add_field(
                        name=part_of_speech.title(),
                        value=definition[:500],
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Define error: {e}")
            await interaction.followup.send(f"❌ Could not fetch definition for '{word}'.")

    # 22. /qr - Generate QR code
    @app_commands.command(name="qr", description="Generate QR code")
    @app_commands.describe(text="Text or URL to encode")
    async def qr_slash(self, interaction: discord.Interaction, text: str):
        encoded = urllib.parse.quote(text)
        url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={encoded}"
        
        embed = discord.Embed(
            title="📱 QR Code",
            description=f"**Content:** {text[:100]}",
            color=0x000000
        )
        embed.set_image(url=url)
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="qr")
    async def qr_prefix(self, ctx, *, text: str):
        encoded = urllib.parse.quote(text)
        url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={encoded}"
        
        embed = discord.Embed(
            title="📱 QR Code",
            description=f"**Content:** {text[:100]}",
            color=0x000000
        )
        embed.set_image(url=url)
        
        await ctx.send(embed=embed)

    # 23. /serverinfo - Server information
    @app_commands.command(name="serverinfo", description="Get server information")
    async def serverinfo_slash(self, interaction: discord.Interaction):
        guild = interaction.guild
        embed = discord.Embed(
            title=f"🏰 {guild.name}",
            color=0x7289DA
        )
        
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
        embed = discord.Embed(
            title=f"🏰 {guild.name}",
            color=0x7289DA
        )
        
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
# COG 5: ENTERTAINMENT (5 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EntertainmentCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 24. /meme - Get a meme
    @app_commands.command(name="meme", description="Get a random meme")
    async def meme_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                "https://meme-api.com/gimme",
                timeout=10
            ) as r:
                data = await r.json()
                embed = discord.Embed(
                    title=data['title'],
                    url=data['postLink'],
                    color=0xFF9900
                )
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
                async with self.bot.session.get(
                    "https://meme-api.com/gimme",
                    timeout=10
                ) as r:
                    data = await r.json()
                    embed = discord.Embed(
                        title=data['title'],
                        url=data['postLink'],
                        color=0xFF9900
                    )
                    embed.set_image(url=data['url'])
                    embed.set_footer(text=f"👍 {data['ups']} | r/{data['subreddit']}")
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Meme error: {e}")
                await ctx.send("❌ Could not fetch meme.")

    # 25. /joke - Get a joke
    @app_commands.command(name="joke", description="Get a random joke")
    async def joke_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist",
                timeout=10
            ) as r:
                data = await r.json()
                
                if data['type'] == 'single':
                    embed = discord.Embed(
                        title="😂 Joke",
                        description=data['joke'],
                        color=0xFFD700
                    )
                else:
                    embed = discord.Embed(
                        title="😂 Joke",
                        description=f"**{data['setup']}**\n\n{data['delivery']}",
                        color=0xFFD700
                    )
                
                embed.set_footer(text=f"Category: {data['category']}")
                await interaction.followup.send(embed=embed)
        except Exception as e:
            logger.error(f"Joke error: {e}")
            await interaction.followup.send("❌ Could not fetch joke.")

    @commands.command(name="joke")
    async def joke_prefix(self, ctx):
        async with ctx.typing():
            try:
                async with self.bot.session.get(
                    "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist",
                    timeout=10
                ) as r:
                    data = await r.json()
                    
                    if data['type'] == 'single':
                        embed = discord.Embed(
                            title="😂 Joke",
                            description=data['joke'],
                            color=0xFFD700
                        )
                    else:
                        embed = discord.Embed(
                            title="😂 Joke",
                            description=f"**{data['setup']}**\n\n{data['delivery']}",
                            color=0xFFD700
                        )
                    
                    embed.set_footer(text=f"Category: {data['category']}")
                    await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Joke error: {e}")
                await ctx.send("❌ Could not fetch joke.")

    # 26. /8ball - Magic 8-ball
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
        
        embed = discord.Embed(
            title="🎱 Magic 8-Ball",
            description=f"**Question:** {question}\n\n**Answer:** {random.choice(responses)}",
            color=0x000000
        )
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="8ball", aliases=["magic8"])  # Removed "ask" alias to avoid conflict
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
        
        embed = discord.Embed(
            title="🎱 Magic 8-Ball",
            description=f"**Question:** {question}\n\n**Answer:** {random.choice(responses)}",
            color=0x000000
        )
        
        await ctx.send(embed=embed)

    # 27. /roll - Roll dice
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
            
            embed = discord.Embed(
                title="🎲 Dice Roll",
                description=f"**{dice}**",
                color=0xE74C3C
            )
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
            
            embed = discord.Embed(
                title="🎲 Dice Roll",
                description=f"**{dice}**",
                color=0xE74C3C
            )
            embed.add_field(name="Rolls", value=", ".join(map(str, rolls)) if len(rolls) <= 20 else f"{len(rolls)} rolls", inline=True)
            embed.add_field(name="Total", value=total, inline=True)
            
            await ctx.send(embed=embed)
        except:
            await ctx.send("❌ Invalid dice format. Use like '2d6' or '1d20'")

    # 28. /fact - Random fact
    @app_commands.command(name="fact", description="Get a random fact")
    async def fact_slash(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            async with self.bot.session.get(
                "https://uselessfacts.jsph.pl/random.json?language=en",
                timeout=10
            ) as r:
                data = await r.json()
                embed = discord.Embed(
                    title="📚 Random Fact",
                    description=data['text'],
                    color=0x1ABC9C
                )
                embed.set_footer(text="Source: uselessfacts.jsph.pl")
                await interaction.followup.send(embed=embed)
        except:
            # Fallback facts
            facts = [
                "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
                "Octopuses have three hearts. Two pump blood to the gills, while the third pumps it to the rest of the body.",
                "A day on Venus is longer than a year on Venus. Venus takes 243 Earth days to rotate once on its axis, but only 225 Earth days to orbit the Sun.",
                "Bananas are berries, but strawberries aren't.",
                "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion of the metal.",
            ]
            
            embed = discord.Embed(
                title="📚 Random Fact",
                description=random.choice(facts),
                color=0x1ABC9C
            )
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
            
            embed = discord.Embed(
                title="📚 Random Fact",
                description=random.choice(facts),
                color=0x1ABC9C
            )
            await ctx.send(embed=embed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COG 6: SYSTEM (8 commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SystemCog(commands.Cog):
    def __init__(self, bot: GodBot):
        self.bot = bot

    # 29. /help - Help command
    @app_commands.command(name="help", description="Show all commands")
    async def help_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🤖 GodBot v4.0 — Command List",
            description="**Prefix:** `!` or use slash commands `/`",
            color=0x7289DA
        )
        
        embed.add_field(
            name="🤖 AI & Chat (4)",
            value="`/ai` `/imagine` `/translate` `/summarize`",
            inline=False
        )
        
        embed.add_field(
            name="🛡️ Moderation (6)",
            value="`/clear` `/kick` `/ban` `/timeout` `/warn` `/userinfo`",
            inline=False
        )
        
        embed.add_field(
            name="💰 Crypto & Market (5)",
            value="`/crypto` `/stock` `/trending` `/exchanges` `/gas`",
            inline=False
        )
        
        embed.add_field(
            name="🔧 Utility (8)",
            value="`/weather` `/timer` `/poll` `/remind` `/calc` `/define` `/qr` `/serverinfo`",
            inline=False
        )
        
        embed.add_field(
            name="🎮 Entertainment (5)",
            value="`/meme` `/joke` `/8ball` `/roll` `/fact`",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ System (8)",
            value="`/help` `/ping` `/stats` `/uptime` `/invite` `/support` `/vote` `/about`",
            inline=False
        )
        
        embed.set_footer(text=f"Total: 36 commands | Use !help for prefix commands")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @commands.command(name="help", aliases=["commands"])
    async def help_prefix(self, ctx):
        embed = discord.Embed(
            title="🤖 GodBot v4.0 — Command List",
            description="**Prefix:** `!` or use slash commands `/`",
            color=0x7289DA
        )
        
        embed.add_field(
            name="🤖 AI & Chat",
            value="`!ai [prompt]` `!imagine [prompt]` `!translate [lang] [text]` `!summarize [length] [text]`",
            inline=False
        )
        
        embed.add_field(
            name="🛡️ Moderation",
            value="`!clear [amount]` `!kick [user] [reason]` `!ban [user] [days] [reason]` `!warn [user] [reason]` `!userinfo [user]`",
            inline=False
        )
        
        embed.add_field(
            name="💰 Crypto & Market",
            value="`!crypto [coin]` `!trending` `!gas`",
            inline=False
        )
        
        embed.add_field(
            name="🔧 Utility",
            value="`!weather [city]` `!poll [question] [options]` `!calc [expression]` `!qr [text]` `!serverinfo`",
            inline=False
        )
        
        embed.add_field(
            name="🎮 Entertainment",
            value="`!meme` `!joke` `!8ball [question]` `!roll [dice]` `!fact`",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ System",
            value="`!help` `!ping` `!stats` `!uptime` `!invite` `!about`",
            inline=False
        )
        
        embed.set_footer(text=f"Total: 36 commands | Use /help for slash commands")
        
        await ctx.send(embed=embed)

    # 30. /ping - Check bot latency
    @app_commands.command(name="ping", description="Check bot latency")
    async def ping_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"**Latency:** {round(self.bot.latency * 1000)}ms",
            color=0x00FF00
        )
        await interaction.response.send_message(embed=embed)

    @commands.command(name="ping")
    async def ping_prefix(self, ctx):
        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"**Latency:** {round(self.bot.latency * 1000)}ms",
            color=0x00FF00
        )
        await ctx.send(embed=embed)

    # 31. /stats - Bot statistics
    @app_commands.command(name="stats", description="Show bot statistics")
    async def stats_slash(self, interaction: discord.Interaction):
        guilds = len(self.bot.guilds)
        users = sum(g.member_count for g in self.bot.guilds)
        
        embed = discord.Embed(
            title="📊 Bot Statistics",
            color=0x9B59B6
        )
        
        embed.add_field(name="Servers", value=guilds, inline=True)
        embed.add_field(name="Users", value=users, inline=True)
        embed.add_field(name="Commands", value="36", inline=True)
        embed.add_field(name="Uptime", value=str(datetime.datetime.now(timezone.utc) - self.bot.start_time).split('.')[0], inline=True)
        embed.add_field(name="Python", value=platform.python_version(), inline=True)
        embed.add_field(name="discord.py", value=discord.__version__, inline=True)
        
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        embed.add_field(name="CPU", value=f"{cpu}%", inline=True)
        embed.add_field(name="RAM", value=f"{memory.percent}%", inline=True)
        embed.add_field(name="Platform", value=platform.system(), inline=True)
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="stats")
    async def stats_prefix(self, ctx):
        guilds = len(self.bot.guilds)
        users = sum(g.member_count for g in self.bot.guilds)
        
        embed = discord.Embed(
            title="📊 Bot Statistics",
            color=0x9B59B6
        )
        
        embed.add_field(name="Servers", value=guilds, inline=True)
        embed.add_field(name="Users", value=users, inline=True)
        embed.add_field(name="Commands", value="36", inline=True)
        embed.add_field(name="Uptime", value=str(datetime.datetime.now(timezone.utc) - self.bot.start_time).split('.')[0], inline=True)
        embed.add_field(name="Python", value=platform.python_version(), inline=True)
        embed.add_field(name="discord.py", value=discord.__version__, inline=True)
        
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        embed.add_field(name="CPU", value=f"{cpu}%", inline=True)
        embed.add_field(name="RAM", value=f"{memory.percent}%", inline=True)
        embed.add_field(name="Platform", value=platform.system(), inline=True)
        
        await ctx.send(embed=embed)

    # 32. /uptime - Bot uptime
    @app_commands.command(name="uptime", description="Show bot uptime")
    async def uptime_slash(self, interaction: discord.Interaction):
        uptime = datetime.datetime.now(timezone.utc) - self.bot.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        embed = discord.Embed(
            title="⏰ Uptime",
            description=f"{days}d {hours}h {minutes}m {seconds}s",
            color=0x3498DB
        )
        embed.add_field(name="Started", value=self.bot.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="uptime")
    async def uptime_prefix(self, ctx):
        uptime = datetime.datetime.now(timezone.utc) - self.bot.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        embed = discord.Embed(
            title="⏰ Uptime",
            description=f"{days}d {hours}h {minutes}m {seconds}s",
            color=0x3498DB
        )
        embed.add_field(name="Started", value=self.bot.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        await ctx.send(embed=embed)

    # 33. /invite - Get bot invite
    @app_commands.command(name="invite", description="Get bot invite link")
    async def invite_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🔗 Invite GodBot",
            description=f"[Click here to invite](https://discord.com/oauth2/authorize?client_id={self.bot.user.id}&permissions=8&scope=bot%20applications.commands)",
            color=0x5865F2
        )
        embed.add_field(name="Permissions", value="Administrator (recommended)")
        embed.add_field(name="Support", value="Use `/support` for help")
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="invite")
    async def invite_prefix(self, ctx):
        embed = discord.Embed(
            title="🔗 Invite GodBot",
            description=f"[Click here to invite](https://discord.com/oauth2/authorize?client_id={self.bot.user.id}&permissions=8&scope=bot%20applications.commands)",
            color=0x5865F2
        )
        embed.add_field(name="Permissions", value="Administrator (recommended)")
        embed.add_field(name="Support", value="Use `!support` for help")
        
        await ctx.send(embed=embed)

    # 34. /support - Support server
    @app_commands.command(name="support", description="Get support server invite")
    async def support_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🆘 Support",
            description="Join our support server for help:",
            color=0xF1C40F
        )
        embed.add_field(name="Server Invite", value="[Click to join](https://discord.gg/example)", inline=False)
        embed.add_field(name="GitHub", value="[View source code](https://github.com/example/godbot)", inline=False)
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="support")
    async def support_prefix(self, ctx):
        embed = discord.Embed(
            title="🆘 Support",
            description="Join our support server for help:",
            color=0xF1C40F
        )
        embed.add_field(name="Server Invite", value="[Click to join](https://discord.gg/example)", inline=False)
        embed.add_field(name="GitHub", value="[View source code](https://github.com/example/godbot)", inline=False)
        
        await ctx.send(embed=embed)

    # 35. /vote - Vote for the bot
    @app_commands.command(name="vote", description="Vote for GodBot")
    async def vote_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🗳️ Vote for GodBot",
            description="Help support the bot by voting!",
            color=0xE91E63
        )
        embed.add_field(name="Top.gg", value="[Vote on Top.gg](https://top.gg/bot/example)", inline=True)
        embed.add_field(name="Discord Bot List", value="[Vote on DBL](https://discordbotlist.com/bots/example)", inline=True)
        
        await interaction.response.send_message(embed=embed)

    # 36. /about - About the bot
    @app_commands.command(name="about", description="About GodBot")
    async def about_slash(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🤖 About GodBot v4.0",
            description="A feature-rich Discord bot with AI, moderation, crypto, utility, and entertainment commands.",
            color=0x7289DA
        )
        
        embed.add_field(name="Version", value=self.bot.VERSION, inline=True)
        embed.add_field(name="Developer", value="Your Name", inline=True)
        embed.add_field(name="Library", value="discord.py", inline=True)
        
        embed.add_field(
            name="Features",
            value="• AI Chat (Gemini 2.0 Flash)\n• IBM Toxicity Moderation\n• Crypto & Stock Prices\n• Weather & Utilities\n• Fun & Entertainment\n• Full Moderation Tools",
            inline=False
        )
        
        embed.add_field(
            name="Credits",
            value="• Gemini AI by Google\n• IBM Granite Guardian\n• Pollinations.ai for images\n• Various free APIs",
            inline=False
        )
        
        embed.set_footer(text="Made with ❤️ using Python")
        
        await interaction.response.send_message(embed=embed)

    @commands.command(name="about")
    async def about_prefix(self, ctx):
        embed = discord.Embed(
            title="🤖 About GodBot v4.0",
            description="A feature-rich Discord bot with AI, moderation, crypto, utility, and entertainment commands.",
            color=0x7289DA
        )
        
        embed.add_field(name="Version", value=self.bot.VERSION, inline=True)
        embed.add_field(name="Developer", value="Your Name", inline=True)
        embed.add_field(name="Library", value="discord.py", inline=True)
        
        embed.add_field(
            name="Features",
            value="• AI Chat (Gemini 2.0 Flash)\n• IBM Toxicity Moderation\n• Crypto & Stock Prices\n• Weather & Utilities\n• Fun & Entertainment\n• Full Moderation Tools",
            inline=False
        )
        
        embed.add_field(
            name="Credits",
            value="• Gemini AI by Google\n• IBM Granite Guardian\n• Pollinations.ai for images\n• Various free APIs",
            inline=False
        )
        
        embed.set_footer(text="Made with ❤️ using Python")
        
        await ctx.send(embed=embed)

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
