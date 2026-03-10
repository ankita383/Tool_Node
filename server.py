from fastmcp import FastMCP

mcp=FastMCP("MathServer")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtracts two numbers."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divides two numbers."""
    if b == 0: return "Error: Division by zero"
    return a / b

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)
