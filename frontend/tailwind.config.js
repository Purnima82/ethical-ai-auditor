/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['Space Mono', 'monospace'],
        sans: ['Syne', 'system-ui', 'sans-serif'],
      },
      colors: {
        dark: {
          bg: '#0a0a0f',
          surface: '#12121a',
          surface2: '#1a1a26',
        },
        accent: {
          purple: '#7c6df0',
          pink: '#e05b8b',
          teal: '#3ecfb0',
          amber: '#f5a623',
        },
      },
    },
  },
  plugins: [],
}
