question_answering_system = '''
You are the host of the show Impact Theory, and your name is Tom Bilyeu.  The description of your show is as follows:
If you’re looking to thrive in uncertain times, achieve unprecedented goals, and improve the most meaningful aspects of your life, then Impact Theory is the show for you. Hosted by Tom Bilyeu, a voracious learner and hyper-successful entrepreneur, the show investigates and analyzes the most useful topics with the world’s most sought-after guests. 
Bilyeu attacks each episode with a clear desire to further evolve the holistic skillset that allowed him to co-found the billion-dollar company Quest Nutrition, generate over half a billion organic views on his content, build a thriving marriage of over 20 years, and quantifiably improve the lives of over 10,000 people through his school, Impact Theory University. 
Bilyeu’s insatiable hunger for knowledge gives the show urgency, relevance, and depth while leaving listeners with the knowledge, tools, and empowerment to take control of their lives and develop true personal power.
'''

question_answering_prompt = '''
Answer the following question by reviewing the blocks of context surrounded by triple back ticks:\n
Question:\n
{question}.\n
```{context}```
'''

test_prompt = '''
Answer the following question by reviewing the blocks of context surrounded by triple back ticks:

Question:

How can one master the art of life?.

```So, if you want to be an incredible musician, one of the things you're going to spend a lot of time on are 
scales. So, you're going to be, once you master that, you master the instrument and the finger movements, I'm 
assuming one's playing guitar in this analogy, and you master all of that stuff, then you can express yourself, 
then you can be creative, then you can, as you're saying, you know, create that art. So, when it comes to the art 
of living, what are the scales? What are the things that people can practice? Obviously, I've read your book, which
is tremendous, Green Lights, for anybody that hasn't read it yet, really amazing. Listen to the audiobook. It is 
unbelievable.

It goes from the intellect down into the body, and that's when it becomes an art. That's an individual practice, I 
think, for everybody. But what we're going to do on the 24th is dive deeper into the sort of the digits, the actual
measurable tools of how to get more satisfaction out of life so you can get into the art of living, which is an 
art, you know, facts and fates. The facts and the science, that's the science of satisfaction. The fate and what 
the world's doing without our doing, whether our hand's on the wheel or not, where that road goes and how to 
navigate it, that becomes the art. But the two are not a contradiction. Now, when I think about the great artists 
and music, especially for somebody living in Austin, seems like a great example.

And what do I care about in life? Yes. I was gonna ask you if you're talking to him about that, because when I 
think about the event that you have coming up, when I think about green lights, when I just think about the concept
of the art of living, it's like, you've got three kids, you're gonna have to teach them the art of living. Like, 
how do you, what is that foundation that you lay for them? Because social media, man, that's, you wanna talk about 
something that'll mess up the art of living real fast, make you self-conscious in a way that's not useful, that 
will shape, at that age, oh my God, that will shape the sense of who you are, which then actually impacts who you 
become. Ooh. It's scary, man. Scary.

I love it. The instilling values, what you're doing, what you did with the book, what you're doing now with the Art
of Living, the event, which I think is really exciting. If you want to tell people when and where to go for that, 
it would be amazing. April 24th at 9 a.m. Pacific. Artoflivingevent.com. You can go there and reserve a spot now. 
It's going to be myself. It's going to be Tony Robbins, Dean Graziosi, Trent Shelton, Mary Ferleo. And we're going 
to get under the hood of Greenlight's approach and get into the process and hopefully share some tools with you 
individually that you can apply in your own life. To one, get on the road to the science of the satisfaction you're
going to have to then get into the art of living.

There are going to be hard times. And I've heard you say something that I think is very powerful, which is never 
see yourself as a victim. And so as we're all going through this life and things are getting difficult and you're 
trying to hold on to that image of what you could be, of what life could be, and you're getting lashed by, you 
know, the reaches of the jungle, but that whether it's religion or just what one ought to do, that you have a very 
clear vision of what it is on the other side to keep you pushing through all that. Now, you have an event coming up
called the Art of Living. Is that what you mean? Oh, that's sure part of it. It's not that you have to be a 
believer in the art to achieve the art of living.```
'''

# If the answer is not provided in the context then respond with: "Answer not found in context."\n