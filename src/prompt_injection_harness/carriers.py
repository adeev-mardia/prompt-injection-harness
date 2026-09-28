"""Sample clean "carrier" documents.

These represent realistic untrusted content that a RAG/agent pipeline
might retrieve: product reviews, support tickets, wiki articles, and
similar. The generator injects payloads into (copies of) these documents
to build a labeled synthetic corpus. None of these documents contain any
injected content on their own -- they are the clean baseline.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class CarrierDocument(BaseModel):
    name: str
    kind: str
    text: str


CARRIER_DOCUMENTS: List[CarrierDocument] = [
    CarrierDocument(
        name="product_review_headphones",
        kind="product_review",
        text=(
            "I bought these wireless headphones last month and have been using "
            "them daily on my commute. The noise cancellation is genuinely "
            "impressive for the price point, and the battery lasts close to "
            "the advertised 30 hours. Pairing with my phone was instant and I "
            "have not had a single dropout. The only downside is the carrying "
            "case feels a bit flimsy, but that is a minor complaint. Overall "
            "I would recommend these to anyone looking for a mid-range pair "
            "that punches above its weight. Comfortable for long listening "
            "sessions too, no ear fatigue even after several hours."
        ),
    ),
    CarrierDocument(
        name="support_ticket_login_issue",
        kind="support_ticket",
        text=(
            "Subject: Unable to log in after password reset\n\n"
            "Hi team, I reset my password yesterday using the 'forgot "
            "password' link, but I'm still getting an 'invalid credentials' "
            "error when I try to sign in on both the mobile app and the web "
            "dashboard. I've double-checked for typos and even tried a fresh "
            "browser in incognito mode. My account email is on file. This "
            "started happening right after the reset email came through. "
            "Could someone check whether the reset actually took effect on "
            "the backend? Happy to hop on a call if that's faster. Thanks in "
            "advance for the help."
        ),
    ),
    CarrierDocument(
        name="wiki_article_photosynthesis",
        kind="wiki_article",
        text=(
            "Photosynthesis is the process by which green plants, algae, and "
            "some bacteria convert light energy, usually from the sun, into "
            "chemical energy stored in glucose. The process takes place "
            "primarily in the chloroplasts of plant cells, using the pigment "
            "chlorophyll to absorb light. Photosynthesis can be summarized by "
            "the reaction: carbon dioxide and water, in the presence of light "
            "energy, produce glucose and oxygen. This process is fundamental "
            "to life on Earth, as it is the primary source of atmospheric "
            "oxygen and forms the base of most food chains."
        ),
    ),
    CarrierDocument(
        name="support_ticket_billing",
        kind="support_ticket",
        text=(
            "Subject: Duplicate charge on my last invoice\n\n"
            "Hello, I noticed two identical charges of $49.00 on my card "
            "statement dated the same day, both labeled with your company "
            "name. I only have one active subscription on my account. Could "
            "you please look into this and refund the duplicate charge if "
            "confirmed? I've attached a screenshot of the statement for "
            "reference. My subscription has been active for about eight "
            "months without any prior billing issues, so this seems like an "
            "isolated glitch. Appreciate a quick resolution."
        ),
    ),
    CarrierDocument(
        name="wiki_article_http",
        kind="wiki_article",
        text=(
            "The Hypertext Transfer Protocol (HTTP) is an application-layer "
            "protocol for transmitting hypermedia documents, such as HTML. It "
            "was designed for communication between web browsers and web "
            "servers, but it can also be used for other purposes. HTTP "
            "follows a classical client-server model, with a client opening a "
            "connection to make a request and waiting for a response. HTTP is "
            "stateless, meaning the server does not retain information "
            "between requests, though cookies allow stateful sessions to be "
            "layered on top of it. Common methods include GET, POST, PUT, "
            "and DELETE."
        ),
    ),
    CarrierDocument(
        name="product_review_blender",
        kind="product_review",
        text=(
            "This blender has completely replaced my old one. It handles "
            "frozen fruit smoothies without any struggle and the motor barely "
            "gets warm even after several minutes of continuous use. Cleanup "
            "is easy since most parts are dishwasher safe. The pulse setting "
            "gives good control for chunkier salsas. It is a little loud, "
            "which is expected for something this powerful, but not louder "
            "than comparable models I've used before. The included recipe "
            "booklet was a nice touch for a beginner like me. Would buy "
            "again."
        ),
    ),
]
