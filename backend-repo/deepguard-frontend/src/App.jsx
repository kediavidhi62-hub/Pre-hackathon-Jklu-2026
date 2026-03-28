import { useState } from 'react'
import NeuralBackground from './components/NeuralBackground'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import UploadCard from './components/UploadCard'
import ResultPanel from './components/ResultPanel'
import Features from './components/Features'
import FAQ from './components/FAQ'
import Footer from './components/Footer'

export default function App() {
  const [result, setResult] = useState(null)

  const handleResult = (data) => {
    setResult(data)
    /* scroll to result */
    setTimeout(() => {
      document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  const handleReset = () => {
    setResult(null)
    setTimeout(() => {
      document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  const scrollToUpload = () => {
    document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Full-screen animated neural network mesh */}
      <NeuralBackground />

      {/* All content above the background */}
      <div className="relative" style={{ zIndex: 1 }}>
        <Navbar />
        <Hero onStartDetecting={scrollToUpload} />

        {/* Upload / Result — swaps based on state */}
        {result ? (
          <ResultPanel result={result} onReset={handleReset} />
        ) : (
          <UploadCard onResult={handleResult} />
        )}

        <Features />
        <FAQ />
        <Footer />
      </div>
    </div>
  )
}
