import os
import time
import datetime
import json



def view_endpoint_deployed_models(endpoint):
    """
    Prints detailed information about models deployed to an Endpoint.
    
    Args:

        endpoint (google.cloud.aiplatform.Endpoint):
            Endpoint hosting the model(s).
    """

    deployed_model_list = endpoint.list_models()

    if len(deployed_model_list) == 0:
        print("No models deployed")
        return

    #

    column_name_list = [
        "DEPLOYED_MODEL_ID",
        "MODEL_ID",
        "DISPLAY_NAME"
    ]

    column_data_list_dict = {
        column_name: []
        for column_name in column_name_list
    }

    for deployed_model in endpoint.list_models():

        column_data_list_dict["DEPLOYED_MODEL_ID"].append(deployed_model.id)
        column_data_list_dict["MODEL_ID"].append(os.path.split(deployed_model.model)[-1])
        column_data_list_dict["DISPLAY_NAME"].append(deployed_model.display_name)

    #

    deployed_model_id_to_idx = {
        deployed_model_id: idx
        for idx, deployed_model_id in enumerate(column_data_list_dict["DEPLOYED_MODEL_ID"])
    }

    column_name_list.append("TRAFFIC_SPLIT")
    column_data_list_dict["TRAFFIC_SPLIT"] = ["-" for _ in range(len(deployed_model_id_to_idx))]

    for deployed_model_id, traffic_split_val in endpoint.gca_resource.traffic_split.items():
        deployed_model_idx = deployed_model_id_to_idx[deployed_model_id]
        column_data_list_dict["TRAFFIC_SPLIT"][deployed_model_idx] = str(traffic_split_val)

    # Compute column sizes

    column_size_dict = {
        column_name: len(column_name)
        for column_name in column_name_list
    }

    for column_name, column_val_list in column_data_list_dict.items():
        for column_val in column_val_list:
            column_size_dict[column_name] = max(column_size_dict[column_name], len(column_val))

    column_fmt_str_dict = {
        column_name: "{:>" + "{:d}".format(column_size) + "s}"
        for column_name, column_size in column_size_dict.items()
    }

    # Print columns

    print("  ".join([
        column_fmt_str_dict[column_name].format(column_name)
        for column_name in column_name_list
    ]))

    num_models = len(column_data_list_dict[column_name_list[0]])
    for model_idx in range(num_models):
        print("  ".join([
            column_fmt_str_dict[column_name].format(column_data_list_dict[column_name][model_idx])
            for column_name in column_name_list
        ]))



def wait_until_model_is_ready(
    endpoint,
    retry_every=15,
    timeout=20 * 60,
    verbose=True
):
    """
    Sends pings to an Endpoint until the hosted model(s) is ready to serve inference. Useful for
    waiting through the scaleup period of autoscale-to-zero models.
    
    Args:

        endpoint (google.cloud.aiplatform.Endpoint):
            Endpoint hosting the model(s).
        
        retry_every (float):
            Time (seconds) to wait between pings sent to the endpoint.
            Default: 15 s.
        
        timeout (float):
            Time (seconds) to wait until a timeout is issued if the model does not become ready.
            If `None` is provided, no timeout will be issued.
            Default: 20 min.

        verbose (bool):
            If `True`, ping response times and status code will be printed to the terminal.
            Default: `True`.

    Returns:

        bool:
            `True` iff the model is ready.
    """

    init_time = time.time()

    while True:

        try:

            resp = endpoint.invoke(
                "/ping",
                body=json.dumps({}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
        
        except:

            if verbose:
                print("Unexpected error")

            return False

        if verbose:

            msg = "{:s} - {:3d}".format(
                datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                resp.status_code
            )

            if resp.status_code == 200:
                msg += " (Model ready)"
            elif resp.status_code == 429:
                msg += " (Model scaleup...)"
            else:
                msg += " (Unexpected status code - Please report this issue)"

            print(msg)

        if resp.status_code == 200: return True

        time.sleep(retry_every)

        curr_time = time.time()
        elapsed_time = curr_time - init_time
        if (timeout is not None) and (elapsed_time >= timeout):

            if verbose:
                print("Timeout exceeded")
                
            return False
