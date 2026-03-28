import { ArrowRight, Play } from 'lucide-react'

export default function Hero({ onStartDetecting }) {
    return (
        <section className="relative pt-32 pb-24 px-6 overflow-hidden">
            <div className="max-w-4xl mx-auto text-center">
                {/* Badge */}
                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-50 border border-blue-100 mb-8 animate-fade-in">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-sm font-medium text-brand-blue">AI-Powered Detection Engine</span>
                </div>

                {/* Headline */}
                <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold leading-tight mb-6 animate-slide-up text-navy">
                    Detect Deepfakes in{' '}
                    <br className="hidden sm:block" />
                    Seconds with{' '}
                    <span className="gradient-text">AI Precision</span>
                </h1>

                {/* Subtext */}
                <p className="text-lg md:text-xl text-soft-gray max-w-2xl mx-auto mb-10 animate-slide-up" style={{ animationDelay: '0.1s' }}>
                    Upload an image or video file and instantly verify authenticity
                    using explainable AI.
                </p>

                {/* CTA Buttons */}
                <div className="flex items-center justify-center animate-slide-up" style={{ animationDelay: '0.2s' }}>
                    <button
                        onClick={onStartDetecting}
                        className="gradient-btn text-white font-semibold px-8 py-3.5 rounded-xl shadow-lg shadow-brand-blue/30 hover:shadow-xl hover:shadow-brand-purple/40 hover:scale-[1.02] transition-all duration-300 flex items-center gap-2"
                    >
                        Start Detecting
                        <ArrowRight className="w-5 h-5" />
                    </button>
                </div>

                {/* Trust line */}
                <p className="mt-10 text-sm text-soft-gray/70 animate-fade-in" style={{ animationDelay: '0.4s' }}>
                    Trusted by 10,000+ journalists, researchers & students worldwide
                </p>
            </div>
        </section>
    )
}
