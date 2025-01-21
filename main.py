import trtllm_c

def main():
    # Initialize parameters
    params = trtllm_c.TlcInitParams()
    params.engine_path = "/engines/rank0.engine"

    # Initialize the executor
    executor = trtllm_c.tlc_init(params)

    # Create a request
    request = trtllm_c.TlcRequest()
    request.tokens = [101, 102, 103]
    request.num_tokens = len(request.tokens)
    request.client_req_id = 1
    request.params = trtllm_c.TlcRequestParams()
    request.params.max_new_tokens = 10
    request.params.temperature = 0.7

    # Enqueue the request
    req_id = trtllm_c.tlc_enqueue_request(executor, request)

    # Await responses
    responses = trtllm_c.tlc_await_responses(executor, timeout_ms=5000)
    for response in responses:
        print(f"Response ID: {response.req_id}, Tokens: {response.tokens}")

    # Shutdown the executor
    trtllm_c.tlc_shutdown(executor)

if __name__ == "__main__":
    main()