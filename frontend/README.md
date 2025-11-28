# Frontend

## UI Framework

This frontend is built on **[Minimal UI](https://github.com/minimal-ui-kit/material-kit-react)** (Material Kit React), a React admin dashboard template.

### Key Resources for UI Development

- **Documentation**: https://docs.minimals.cc/
- **Repository**: https://github.com/minimal-ui-kit/material-kit-react
- **Tech Stack**: React, Material-UI (MUI v5), Vite, TypeScript

### Architecture Highlights

- **Theme System**: Located in `src/theme/core/` - uses custom utilities like `varAlpha()`, `hexToRgbChannel()`, and color channel variables
- **Settings Drawer**: `src/components/settings/drawer/` - handles theme customization (light/dark mode, 6 primary color presets, navigation layouts)
- **Layouts**: Multiple layout patterns available in `src/layouts/` (auth-centered, auth-split, dashboard, simple)
- **Navigation**: Supports vertical, horizontal, and mini navigation variants

When making UI changes, refer to the Minimal UI documentation for component patterns and theming conventions.

---

## Prerequisites

- Node.js 20.x (Recommended)

## Installation

**Using Yarn (Recommended)**

```sh
yarn install
yarn dev
```

**Using Npm**

```sh
npm i
npm run dev
```

## Build

```sh
yarn build
# or
npm run build
```
