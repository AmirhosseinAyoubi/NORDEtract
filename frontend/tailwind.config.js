/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        nordea: {
          blue: '#007aff',
          'blue-light': '#4da3ff',
          'blue-dark': '#0056cc'
        }
      }
    },
  },
  plugins: [],
}
