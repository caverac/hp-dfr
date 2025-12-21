import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const config: Config = {
  title: "Adaptive hp-DFR",
  tagline: "Goal-Oriented Deep Fourier Residual Methods with Adaptive hp-Refinement",
  favicon: "img/favicon.ico",

  url: "https://caverac.github.io",
  baseUrl: "/hp-dfr/",

  organizationName: "caverac",
  projectName: "hp-dfr",

  onBrokenLinks: "throw",

  // Enable Mermaid diagrams
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  themes: ["@docusaurus/theme-mermaid"],

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl: "https://github.com/caverac/hp-dfr/tree/main/packages/docs/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
      type: "text/css",
      integrity: "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
      crossorigin: "anonymous",
    },
  ],

  themeConfig: {
    image: "img/social-card.png",
    navbar: {
      title: "Adaptive hp-DFR",
      logo: {
        alt: "Adaptive hp-DFR Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          href: "https://github.com/caverac/hp-dfr",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Introduction",
              to: "/docs/intro",
            },
            {
              label: "Getting Started",
              to: "/docs/getting-started/installation",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "MATHMODE Group",
              href: "https://www.mathmode.science/",
            },
            {
              label: "Original DFR Paper",
              href: "https://arxiv.org/abs/2210.14129",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/caverac/hp-dfr",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Carlos Vera-Ciro. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["python", "bash", "latex"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
