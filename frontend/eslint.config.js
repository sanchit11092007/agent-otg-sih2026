import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // Ignore generated/cached directories.  Some Windows installs expose a
  // read-protected .pytest_cache through a junction, which otherwise causes
  // eslint to fail before it examines any source files.
  globalIgnores(['dist', 'node_modules', '.pytest_cache', '**/.pytest_cache/**']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    files: ['vite.config.js'],
    languageOptions: { globals: globals.node },
  },
])
