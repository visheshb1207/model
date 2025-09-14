# evaluate assumptions
Detector = AssumptionDetector()
results = Detector.detect(context,response)
assumption = results['confidence']


# evaluate instructions
res = evaluate_instruction_following(system_req, response)
instruction_score = res['overall_score']