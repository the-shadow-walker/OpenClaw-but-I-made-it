"""
Base Agent Class
All specialized agents inherit from this
Uses Ollama HTTP API for reliable model interaction
"""

import asyncio
import requests
import json
import time
from typing import Optional, List
from core import Message, MessageType, Task, AgentType, message_bus, memory


class BaseAgent:
    """Base class for all agents - uses Ollama HTTP API"""
    
    def __init__(
        self, 
        agent_id: str,
        agent_type: AgentType,
        model_name: str,
        system_prompt: str,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.ollama_base_url = ollama_base_url
        
        # Subscribe to message bus
        self.message_queue = message_bus.subscribe(agent_id)
        
        # Status
        self.is_running = False
        self.current_task: Optional[Task] = None
        
        print(f"🤖 Initialized {agent_id} ({agent_type.value}) using {model_name}")
    
    async def send_message(
        self,
        to_agent: str,
        msg_type: MessageType,
        content: str,
        task_id: Optional[str] = None
    ):
        """Send a message to another agent"""
        message = Message(
            id="",
            timestamp="",
            from_agent=self.agent_id,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            task_id=task_id
        )
        await message_bus.publish(message)
    
    async def process_message(self, message: Message):
        """Process incoming message - override in subclasses"""
        pass
    
    async def execute_task(self, task: Task) -> str:
        """Execute a task - override in subclasses"""
        pass
    
    async def query_llm(self, prompt: str, system: Optional[str] = None, stream: bool = True) -> str:
        """Query the LLM using Ollama HTTP API"""
        
        # Prepare full prompt with system message
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        # Debug info
        print(f"      🔧 [DEBUG] Calling Ollama HTTP API...")
        print(f"      🔧 [DEBUG] Model: {self.model_name}")
        print(f"      🔧 [DEBUG] Prompt length: {len(full_prompt)} chars")
        
        try:
            start_time = time.time()
            
            def _generate():
                print(f"      🔧 [DEBUG] POST {self.ollama_base_url}/api/generate")
                
                # Prepare request payload
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": stream,
                    "keep_alive": 300,   # stay warm for 5 min — explicit unload handles transitions
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048,
                    }
                }
                
                # Make request
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    stream=stream,
                    timeout=120
                )
                response.raise_for_status()
                
                full_response = ""
                token_count = 0
                char_count = 0
                
                if stream:
                    # Stream response line by line
                    print(f"      🔧 [DEBUG] Streaming response...")
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    chunk = data['response']
                                    full_response += chunk
                                    token_count += 1
                                    
                                    # Print with word wrapping
                                    for char in chunk:
                                        print(char, end="", flush=True)
                                        char_count += 1
                                        
                                        if char == '\n':
                                            print("         ", end="", flush=True)
                                            char_count = 0
                                        elif char_count > 80 and char == ' ':
                                            print("\n         ", end="", flush=True)
                                            char_count = 0
                                
                                # Check if done
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    # Non-streaming response
                    data = response.json()
                    full_response = data.get('response', '')
                    token_count = len(full_response.split())
                
                # Final stats
                total_time = time.time() - start_time
                avg_speed = token_count / total_time if total_time > 0 else 0
                print(f"\n      🔧 [DEBUG] Generation complete: {token_count} tokens in {total_time:.1f}s ({avg_speed:.1f} tok/s)")
                
                return full_response.strip()
            
            # Run in thread to not block asyncio
            full_response = await asyncio.to_thread(_generate)
            
            print()
            print(f"      ✓ [{self.agent_id}] complete\n")
            return full_response
                
        except requests.exceptions.Timeout:
            print(f"\n      ❌ Timeout after 120s\n")
            return "Error: Model timeout"
        except requests.exceptions.ConnectionError:
            print(f"\n      ❌ Connection error - is Ollama running?\n")
            return "Error: Cannot connect to Ollama"
        except Exception as e:
            print(f"\n      ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            return f"Error querying LLM: {e}"
    
    async def start(self):
        """Start the agent's message processing loop"""
        self.is_running = True
        memory.set_agent_status(self.agent_id, "idle")
        
        while self.is_running:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=0.5
                )
                await self.process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ {self.agent_id} error: {e}")
    
    async def stop(self):
        """Stop the agent and clean up resources"""
        self.is_running = False
        memory.set_agent_status(self.agent_id, "stopped")
        
        import gc
        gc.collect()
    
    def __repr__(self):
        return f"{self.agent_type.value}:{self.agent_id}({self.model_name})"
