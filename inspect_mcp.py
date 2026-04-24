"""Quick local inspector: dumps everything the OpenBB MCP server exposes."""
import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

OPENBB_BIN = Path(__file__).resolve().parent / "environ" / "bin" / "openbb-mcp"


async def inspect(args: list[str], label: str):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  command: openbb-mcp {' '.join(args)}")
    print(f"{'=' * 70}\n")

    client = MultiServerMCPClient({
        "openbb": {
            "transport": "stdio",
            "command": str(OPENBB_BIN),
            "args": args,
        }
    })

    # Tools
    try:
        tools = await client.get_tools()
        print(f"TOOLS ({len(tools)}):")
        for t in tools:
            desc = (t.description or "").strip().split("\n")[0][:110]
            print(f"  • {t.name}")
            print(f"      {desc}")
    except Exception as e:
        print(f"TOOLS: error — {e}")

    # Prompts
    try:
        async with client.session("openbb") as session:
            prompts_resp = await session.list_prompts()
            prompts = prompts_resp.prompts
            print(f"\nPROMPTS ({len(prompts)}):")
            for p in prompts:
                args_info = ""
                if p.arguments:
                    arg_names = [a.name for a in p.arguments]
                    args_info = f"  args: {arg_names}"
                print(f"  • {p.name}{args_info}")
                if p.description:
                    print(f"      {p.description[:110]}")
    except Exception as e:
        print(f"\nPROMPTS: error — {e}")

    # Resources
    try:
        async with client.session("openbb") as session:
            res_resp = await session.list_resources()
            resources = res_resp.resources
            print(f"\nRESOURCES ({len(resources)}):")
            for r in resources[:20]:
                print(f"  • {r.uri}")
                if r.name and r.name != str(r.uri):
                    print(f"      name: {r.name}")
            if len(resources) > 20:
                print(f"  ... and {len(resources) - 20} more")
    except Exception as e:
        print(f"\nRESOURCES: error — {e}")


async def main():
    # With admin filter
    await inspect(["--transport", "stdio", "--default-categories", "admin"],
                  "WITH --default-categories admin")

    # Without any filter
    await inspect(["--transport", "stdio"],
                  "WITHOUT filter (default startup)")


if __name__ == "__main__":
    asyncio.run(main())