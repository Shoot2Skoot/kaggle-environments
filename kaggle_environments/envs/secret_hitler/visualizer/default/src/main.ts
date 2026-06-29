import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { secretHitlerTransformer, getStepLabel, getStepDescription } from './transformer';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'secret_hitler',
    renderer: renderer as any,
    ui: 'inline',
    transformer: (replay: any) => secretHitlerTransformer(replay),
    getStepLabel: (step: any) => getStepLabel(step),
    getStepDescription: (step: any) => getStepDescription(step),
  })
);
