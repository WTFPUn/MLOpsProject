import gradio as gr

ddata = {
    "Category A": {"info": "Info about Category A"},
    "Category B": {"info": "Info about Category B"},
    "Category C": {"info": "Info about Category C"},
}

def query_groups(filter_value):
    if filter_value == "All":
        return list(data.keys())
    else:
        return [k for k in data if filter_value in k]

def show_group_info(group_name):
    return data.get(group_name, {}).get("info", "No info available")

with gr.Blocks() as demo:
    state = gr.State([])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Filter")
            dropdown = gr.Dropdown(choices=["All", "Category A", "Category B", "Category C"], label="Select category")
            query_btn = gr.Button("Query")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Group Cards")
            group_cards = gr.Radio(choices=[], label="Groups", interactive=True)

        with gr.Column(scale=2):
            gr.Markdown("### Group Info")
            group_info = gr.Textbox(label="Group Details", lines=5)

    def update_group_cards(filter_value):
        groups = query_groups(filter_value)
        return gr.update(choices=groups, value=None), groups

    query_btn.click(fn=update_group_cards, inputs=dropdown, outputs=[group_cards, state])
    group_cards.change(fn=show_group_info, inputs=group_cards, outputs=group_info)

demo.launch()
