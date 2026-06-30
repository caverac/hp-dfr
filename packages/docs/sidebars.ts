import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Our Research',
      items: [
        'preprint/index',
        'preprint/literature-review',
        'preprint/phase3-goal-oriented',
      ],
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Theory',
      items: [
        'theory/pinns',
        'theory/dfr',
        'theory/comparison',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/tensorflow',
        'api/jax',
        'api/pytorch',
      ],
    },
    {
      type: 'category',
      label: 'Background',
      items: [
        'paper/summary',
      ],
    },
  ],
};

export default sidebars;
