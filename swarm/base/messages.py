"""
Minimal langchain_core mock for offline development
"""

# Messages module
class messages:
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = 'human'
    
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = 'system'
    
    class AIMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = 'ai'


# For direct imports
HumanMessage = messages.HumanMessage
SystemMessage = messages.SystemMessage
AIMessage = messages.AIMessage
