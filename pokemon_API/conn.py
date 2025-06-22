import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json 
    import requests
    return (requests,)


@app.cell
def _(requests):
    response = requests.get("https://pokeapi.co/api/v2/pokemon")

    # TODO: Create a dataframe containing pokemon name and ID (primary key)
    def get_pokemons_names():
        names = []
        url = []
        for data in response.json()['results']:
            names.append(data['name'])
    return


@app.function
# TODO: Create a list of important data for given pokemon ID 
# Important data: abilities (ID ref), height, weight, stats, types (ID ref)

def get_pokemon_data(id):
    return None


if __name__ == "__main__":
    app.run()
