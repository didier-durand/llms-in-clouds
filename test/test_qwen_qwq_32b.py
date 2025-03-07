import unittest
import time

import openai
from openai.types.chat.chat_completion import Choice

BACKEND_DNS = "<put the DNS of your AWS ALB own here>"
BACKEND_URL = f"http://{BACKEND_DNS}:30000/v1" # noqa

class TestQwB32B(unittest.TestCase):

    """
    some prompting tests including reasoning questions for Qwen QwQ-32B
    """

    def setUp(self):
        self.client = openai.Client(base_url=BACKEND_URL)

    def prompt_qwq_32b (self,  prompt: str = ""):
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=0,
            )
        except Exception as e:  # noqa pylint: disable=W0718
            print(f"exception  latency: {time.perf_counter() - start:.2f}s")
            raise e
        print(f"inference latency: {time.perf_counter() - start:.2f}s")
        print(response)
        self.assertIsNotNone(response)
        self.assertEqual(1,len(response.choices))
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        choice: Choice = response.choices[0]
        self.assertEqual("stop",choice.finish_reason)
        self.assertEqual("assistant",choice.message.role)
        print(f"response (tokens - input: {prompt_tokens:,} / output: {completion_tokens:,}): {choice.message.content}")
        return prompt_tokens, completion_tokens, choice.message.content

    def test_who_are_you(self):
        _, _, response = self.prompt_qwq_32b("who are you ?")
        self.assertTrue("Qwen" in response)
        self.assertTrue("Alibaba" in response)

    def test_french_prompt(self):
        _ , _, response = self.prompt_qwq_32b("Quelle est la capitale de la France ? Combien a-t-elle d'habitants ? "
                                              "Merci de répondre en français")
        self.assertTrue("Paris" in response)
        self.assertTrue("France" in response)
        self.assertTrue("millions" in response)
        self.assertTrue("habitants" in response)


    def test_reasoning_0(self):
        _ , _, response = self.prompt_qwq_32b("How many letters R in word 'strawberry' ? (please, be succinct on your reasoning "
                                              "in the answer)")
        self.assertGreater(len(response),0)

    def test_logical_reasoning_1(self):
        """
        Answer: CABDE. Putting the first three in order, A finished in front of B but behind C, so CAB.
        Then, we know D finished before B, so CABD. We know E finished after D, so CABDE.
        """
        _ , _, response = self.prompt_qwq_32b("Five people were eating apples, A finished before B, but behind C. "
                                              "D finished before E, but behind B. What was the finishing order?")
        self.assertGreater(len(response),0)

    def test_logical_reasoning_2(self):
        """
        40 socks. If he takes out 38 socks (adding the two biggest amounts, 21 and 17), although it is very unlikely,
        it is possible they could all be blue and red. To make 100 percent certain that he also has a pair of black
        socks he must take out a further two socks.
        """
        _ , _, response = self.prompt_qwq_32b("A man has 53 socks in his drawer: 21 identical blue, 15 identical black and "
                                              "17 identical red. The lights are out, and he is completely in the dark. "
                                              "How many socks must he take out to make 100 percent certain he has at least "
                                              "one pair of black socks?")
        self.assertGreater(len(response), 0)

    def test_logical_reasoning_3(self):

        """
        Eleven. Because Lisa lost three games to Susan, she lost $3 ($1 per game). So, she had to win back that $3 with
        three more games, then win another five games to win $5.
        """
        _ , _, response = self.prompt_qwq_32b("Susan and Lisa decided to play tennis against each other. They bet $1 on each game "
                                              "they played. Susan won three bets and Lisa won $5. How many games did they play?")
        self.assertGreater(len(response), 0)
