# Tricky common tasks (under construction)

There are several common tasks which turn out to be surprisingly tricky, even for experience users.  This notebook is intended to serve as a repository for effective methods to perform these common tasks:

- get the max finite value in a list: `see oatpy.barcode.max_finite_value`
- filter rows of the barcode dataframe: see the documentation for Pandas
- saving a plotly image: use `import plotly.io as pio`
- remove perspective effect: use orthographic projection, as demonstrated in the `_plot_surfaces.ipynb`


# set aspect ratio
```
fig.update_layout( 
    title=dict(text="A torus curve"),
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1), # this controls aspect ratio and zoom
        xaxis = dict(range=[-1.5, 1.5],),
        yaxis = dict(range=[-1.5, 1.5],),
        zaxis = dict(range=[-1.5, 1.5],),                
    ),
    width=800, 
    height=800,
    template="plotly_dark",
)    
```