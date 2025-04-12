import asyncio
from typing import Dict, Any, AsyncGenerator
from fastmcp import FastMCP, Context
from ollama import AsyncClient, GenerateResponse, ListResponse
import httpx

# Create the MCP server
mcp = FastMCP("Ollama MCP Server")

# Initialize Ollama client with custom timeout
ollama_client = AsyncClient(timeout=httpx.Timeout(600.0))  # 10 minute timeout


@mcp.tool()
async def generate_text(model: str, prompt: str, ctx: Context) -> Dict[str, Any]:
    """Generate text using Ollama model with progress reporting.

    Args:
        prompt: The text prompt to generate from
        ctx: MCP context for progress reporting

    Returns:
        Dict containing the generated text and metadata
    """
    # Report initial progress
    await ctx.info("Starting text generation...")

    try:
        response = await ollama_client.generate(model=model, prompt=prompt, stream=True)
        chunks = [chunk["response"] async for chunk in response if "response" in chunk]
        full_response = "".join(chunks)
        await ctx.info("Text generation completed successfully")
        return {
            "text": full_response,
            "status": "success"
        }

    except httpx.TimeoutException as e:
        await ctx.error(f"Request timed out: {str(e)}")
        return {
            "error": "Request timed out. The model may be taking too long to respond. Please try again with a shorter prompt or a different model.",
            "status": "error"
        }
    except Exception as e:
        await ctx.error(f"Error during text generation: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }


@mcp.tool()
async def list_models(ctx: Context) -> Dict[str, Any]:
    """List available Ollama models.

    Args:
        ctx: MCP context for logging

    Returns:
        Dict containing list of available models
    """
    try:
        await ctx.info("Fetching available models...")
        response: ListResponse = await ollama_client.list()
        models = response.dict()
        await ctx.info(f"Found {len(models.get('models', []))} models")
        return models
    except Exception as e:
        await ctx.error(f"Error listing models: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    mcp.run()
