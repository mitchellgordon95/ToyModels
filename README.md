# Toy Models of Superposition

An interactive web application for exploring toy models of superposition, based on the Anthropic paper ["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html).

## Live Demo

Visit the live demo at: https://mitchellgordon95.github.io/ToyModels/

## Features

- Interactive visualization of weight matrices and feature superposition
- Adjustable model parameters (dimensions, sparsity, importance decay)
- Preset configurations for common scenarios
- Real-time training with AdamW optimizer
- Visualizations of feature interference and bias terms

## Setup GitHub Pages

To host this on GitHub Pages:

1. Go to your repository settings: https://github.com/mitchellgordon95/ToyModels/settings
2. Scroll down to the "Pages" section
3. Under "Source", select "Deploy from a branch"
4. Choose "main" branch and "/ (root)" folder
5. Click "Save"

Your site will be available at: https://mitchellgordon95.github.io/ToyModels/

## Local Development

Simply open `index.html` in a web browser, or use a local server:

```bash
python -m http.server 8000
# Then visit http://localhost:8000
```

## Technology

Built with vanilla JavaScript, HTML5 Canvas, and CSS - no external dependencies required.