from typing import List, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        # Initialize any necessary attributes
        pass

    async def on_startup(self):
        # Set up any necessary startup procedures
        print("Pipeline started")

    async def on_shutdown(self):
        # Perform any necessary cleanup
        print("Pipeline stopped")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where the main processing happens
        print("Received message:", user_message)
        return "hello"
