!!! tip "Running the async examples"
    The examples on this page use `await` at the top level. That works as is in Jupyter, IPython and other notebooks. In a plain Python script, put the code in an `async` function and run it with `asyncio.run`:

    ```python
    import asyncio


    async def main(): ...  # the example code, including its `await` lines


    asyncio.run(main())
    ```

    Metrics also have synchronous forms (`metric.score(...)`, `metric.batch_score(...)`) that you can call directly from a script.
