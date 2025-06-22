import discord
import os
from discord.ext import commands, tasks
from datetime import datetime, timezone

intents = discord.Intents.default()
intents.members = True
intents.message_content = True  
bot = commands.Bot(command_prefix='!', intents=intents)

GUILD_ID = 1014861739780755519
PROMOTION_CHANNEL_ID = 1020680595757596802

CHECKED_ROLE_IDS = {
    1020682173759627265,  
    1055504940019232870,  
    1037479415472459886,  
}

ROLE_THRESHOLDS = [
    (0,   1382042066468737164),  # Recruit - 1 Month
    (90,   1382044635681259640),  # Squire - 3 Months
    (180,  1382047496343388240),  # Knight - 6 Months
    (365,  1382047626417143898),  # Champion - 1 Year
    (730,  1382061051142602822),  # Veteran - 2 Years
    (1095, 1382058816878678116),  # Warlord - 3 Years
    (1460, 1382066528421281802),  # Elder - 4 Years
    (1825, 1382071830797615114),  # Legend - 5 Years
]

START_DATE = datetime(2023, 12, 30, tzinfo=timezone.utc)


@bot.event
async def on_ready():
    print(f'{bot.user} is online and watching for promotions.')
    update_roles.start()

@bot.event
async def on_member_join(member):
    guild = bot.get_guild(GUILD_ID)
    recruit_role = guild.get_role(1382042066468737164)
    if recruit_role:
        await member.add_roles(recruit_role)

@tasks.loop(hours=24)
async def update_roles():
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        return

    channel = guild.get_channel(PROMOTION_CHANNEL_ID)
    if not channel:
        return

    now = datetime.now(timezone.utc)
    days_since_start = (now - START_DATE).days

    checked_roles = [guild.get_role(rid) for rid in CHECKED_ROLE_IDS if guild.get_role(rid) is not None]
    promotion_roles = [guild.get_role(rid) for _, rid in ROLE_THRESHOLDS if guild.get_role(rid) is not None]

    for member in guild.members:
        if member.bot:
            continue

        has_any_checked_role = any(role in member.roles for role in checked_roles)

        if not has_any_checked_role:
            roles_to_remove = [r for r in promotion_roles if r in member.roles]
            if roles_to_remove:
                print(f"Attempting to remove these roles from {member.name}: {[r.name for r in roles_to_remove]}")
                try:
                    await member.remove_roles(*roles_to_remove)
                    print(f"Successfully removed promotion roles from {member.name} because they lack required roles.")
                except discord.Forbidden:
                    print(f"❌ Missing permissions to remove roles from {member.name}")
                except Exception as e:
                    print(f"⚠️ Error removing roles from {member.name}: {e}")
            else:
                print(f"No promotion roles to remove from {member.name}")
            continue

        now = datetime.now(timezone.utc)
        member_joined_at = member.joined_at or now
        effective_start = max(member_joined_at, START_DATE)
        days_since_effective_start = (now - effective_start).days

        role_to_assign = None
        for threshold_days, role_id in reversed(ROLE_THRESHOLDS):
            if days_since_effective_start >= threshold_days:
                role_to_assign = guild.get_role(role_id)
                break

        if role_to_assign is None:
            continue

        roles_to_remove = [r for r in promotion_roles if r in member.roles and r != role_to_assign]

        try:
            if roles_to_remove:
                await member.remove_roles(*roles_to_remove)
            if role_to_assign not in member.roles:
                await member.add_roles(role_to_assign)
                await channel.send(f"🎉 {member.mention} has been promoted to **{role_to_assign.name}**! You are less of a scrub now , BOBER KURWA!")
        except discord.Forbidden:
            print(f"❌ Missing permissions to assign {role_to_assign.name} to {member.name}")
        except Exception as e:
            print(f"⚠️ Error assigning {role_to_assign.name} to {member.name}: {e}")

@bot.command()
async def test_promotion(ctx):
    channel = bot.get_channel(PROMOTION_CHANNEL_ID)
    if channel:
        try:
            await channel.send("✅ Promotion message test successful! <:bobrkurva:1319803021974310942>")
            await ctx.send("✅ Sent test message to promotion channel.")
        except Exception as e:
            await ctx.send(f"❌ Failed to send message: {e}")
    else:
        await ctx.send("❌ Could not find the promotion channel.")


bot.run(os.environ['DISCORD_TOKEN'])