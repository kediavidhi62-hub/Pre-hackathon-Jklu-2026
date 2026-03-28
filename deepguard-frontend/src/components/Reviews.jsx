import { useState } from 'react'
import { ChevronLeft, ChevronRight, Star } from 'lucide-react'

const REVIEWS = [
    {
        quote: 'I almost published a clip that went viral on social media — until this tool flagged it as manipulated. DeepGuard gave me a breakdown in seconds, and I used the detailed report in my editor briefing. That report saved me from a huge mistake.',
        name: 'Jacob Morris',
        role: 'Investigative Journalist',
        rating: 5,
    },
    {
        quote: 'As a professor teaching media ethics, I use DeepGuard in my classroom to demonstrate real-time deepfake detection. The explainable AI breakdown is perfect for helping students understand how manipulation works.',
        name: 'Dr. Priya Sharma',
        role: 'Professor, Media Studies',
        rating: 5,
    },
    {
        quote: 'Our fact-checking team processes hundreds of tips daily. DeepGuard has become our go-to tool for quick verification. The batch upload feature saves us hours every week. Incredibly reliable.',
        name: 'Alex Chen',
        role: 'Lead Fact-Checker, NewsVerify',
        rating: 5,
    },
    {
        quote: 'I was receiving manipulated images being used against me online. DeepGuard helped me prove they were AI-generated and I was able to take legal action. This tool genuinely helped protect me.',
        name: 'Sarah Thompson',
        role: 'Digital Rights Advocate',
        rating: 5,
    },
]

export default function Reviews() {
    const [current, setCurrent] = useState(0)

    const prev = () => setCurrent((c) => (c === 0 ? REVIEWS.length - 1 : c - 1))
    const next = () => setCurrent((c) => (c === REVIEWS.length - 1 ? 0 : c + 1))

    const review = REVIEWS[current]

    return (
        <section className="py-20 px-6">
            <div className="max-w-2xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        What People <span className="gradient-text">Say</span>
                    </h2>
                    <p className="text-soft-gray">
                        Trusted by journalists, researchers, and students worldwide.
                    </p>
                </div>

                {/* Review Card + Arrows */}
                <div className="relative flex items-center justify-center">
                    {/* Left Arrow */}
                    <button
                        onClick={prev}
                        className="absolute -left-4 md:-left-14 w-10 h-10 rounded-full border border-gray-200 bg-white shadow-sm flex items-center justify-center text-soft-gray hover:text-navy hover:border-brand-blue transition-all z-10"
                    >
                        <ChevronLeft className="w-5 h-5" />
                    </button>

                    {/* Card */}
                    <div className="bg-white rounded-2xl border border-gray-100 shadow-lg p-8 md:p-10 text-center max-w-lg mx-auto">
                        <p className="text-navy italic leading-relaxed mb-6">
                            "{review.quote}"
                        </p>
                        <p className="font-bold text-navy">{review.name}</p>
                        <p className="text-soft-gray text-sm mb-3">{review.role}</p>
                        {/* Stars */}
                        <div className="flex items-center justify-center gap-1 mb-1">
                            {Array.from({ length: review.rating }).map((_, i) => (
                                <Star key={i} className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                            ))}
                        </div>
                        <p className="text-sm font-semibold text-navy">{review.rating}.0</p>
                    </div>

                    {/* Right Arrow */}
                    <button
                        onClick={next}
                        className="absolute -right-4 md:-right-14 w-10 h-10 rounded-full border border-gray-200 bg-white shadow-sm flex items-center justify-center text-soft-gray hover:text-navy hover:border-brand-blue transition-all z-10"
                    >
                        <ChevronRight className="w-5 h-5" />
                    </button>
                </div>

                {/* Dots */}
                <div className="flex items-center justify-center gap-2 mt-6">
                    {REVIEWS.map((_, i) => (
                        <button
                            key={i}
                            onClick={() => setCurrent(i)}
                            className={`w-2.5 h-2.5 rounded-full transition-all ${i === current ? 'bg-brand-blue scale-125' : 'bg-gray-300 hover:bg-gray-400'
                                }`}
                        />
                    ))}
                </div>
            </div>
        </section>
    )
}
