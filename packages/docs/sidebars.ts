import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "intro",
    {
      type: "category",
      label: "Getting Started",
      items: ["getting-started/installation", "getting-started/quick-start"],
    },
    {
      type: "category",
      label: "Theory",
      items: ["theory/pinns", "theory/dfr", "theory/comparison"],
    },
    {
      type: "category",
      label: "Our Research",
      items: ["preprint/index", "preprint/phase3-goal-oriented", "preprint/literature-review"],
    },
    {
      type: "category",
      label: "API Reference",
      items: ["api/tensorflow", "api/jax", "api/pytorch"],
    },
    {
      type: "category",
      label: "Reference",
      items: ["paper/summary"],
    },
  ],
};

export default sidebars;
