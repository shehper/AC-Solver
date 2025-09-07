from ac_solver.envs.utils import simplify_presentation


def concatenate_relators(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by r_i r_j^{sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 0 or 1, but not equal to i.
    sign: +1 or -1, whether to invert r_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with r_i r_j^{sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.
    """
    assert all(
        [
            i in [0, 1],
            j in [0, 1],
            i == 1 - j,
        ]
    ), f"expect i and j to be 0 or 1 and i != j; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    # get r_i
    presentation = presentation.copy()
    relator1 = presentation[i * max_relator_length : (i + 1) * max_relator_length]

    # get r_j or r_j^{-1} depending on sign
    # TODO: really need to understand this
    if sign == 1:
        relator2 = presentation[j * max_relator_length : (j + 1) * max_relator_length]
    elif j:
        relator2 = -presentation[
            (j + 1) * max_relator_length - 1 : j * max_relator_length - 1 : -1
        ]
    else:
        relator2 = -presentation[max_relator_length - 1 :: -1]

    relator1_nonzero = relator1[relator1 != 0]
    relator2_nonzero = relator2[relator2 != 0]

    len1 = len(relator1_nonzero)
    len2 = len(relator2_nonzero)

    acc = 0
    while (
        acc < min(len1, len2) and relator1_nonzero[-1 - acc] == -relator2_nonzero[acc]
    ):
        acc += 1

    new_size = len1 + len2 - 2 * acc

    if new_size <= max_relator_length:
        lengths[i] = new_size
        presentation[i * max_relator_length : i * max_relator_length + len1 - acc] = (
            relator1_nonzero[: len1 - acc]
        )
        presentation[
            i * max_relator_length + len1 - acc : i * max_relator_length + new_size
        ] = relator2_nonzero[acc:]
        presentation[
            i * max_relator_length + new_size : (i + 1) * max_relator_length
        ] = 0

    return presentation, lengths


def conjugate(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by x_j^{sign} r_i x_j^{-sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 1 or 2, index of the generator to conjugate with.
    sign: +1 or -1, whether to invert x_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with x_j^{sign} r_i x_j^{-sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.

    """
    # TODO: perhaps i and j should be more uniformly both in [0, 1].
    assert all(
        [i in [0, 1], j in [1, 2]]
    ), f"expect i to be 0 and 1 and j to be 1 or 2; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    presentation = presentation.copy()
    relator = presentation[i * max_relator_length : (i + 1) * max_relator_length]
    relator_nonzero = relator[relator.nonzero()]
    relator_size = len(relator_nonzero)

    # get the generator that is to be appended on the left
    generator = sign * j

    # TODO: again here, it will be good to use simplify_relator

    # check whether we will need to cancel any generators at the beginning and at the end
    start_cancel = 1 if relator_nonzero[0] == -generator else 0
    end_cancel = 1 if relator_nonzero[-1] == generator else 0

    # get the size of the resultant relator after cancellation
    new_size = relator_size + 2 - 2 * (start_cancel + end_cancel)

    # update lengths and presentation
    if new_size <= max_relator_length:
        lengths = lengths.copy()
        lengths[i] = new_size

        presentation[
            i * max_relator_length
            + 1
            - start_cancel : i * max_relator_length
            + 1
            + relator_size
            - 2 * start_cancel
            - end_cancel
        ] = relator_nonzero[start_cancel : relator_size - end_cancel]

        if not start_cancel:
            presentation[i * max_relator_length] = generator

        if not end_cancel:
            presentation[
                i * max_relator_length + relator_size + 1 - 2 * start_cancel
            ] = -generator

        if start_cancel and end_cancel:
            presentation[
                i * max_relator_length
                + new_size : i * max_relator_length
                + new_size
                + 2
            ] = 0

    return presentation, lengths


def ACMove(move_id, presentation, max_relator_length, lengths, cyclical=True):
    """
    Applies an AC move (concatenation or conjugation) to a presentation and returns the resultant presentation.
    The move to apply and the relator it is applied to are decided by move_id.

    Parameters:
    move_id: An int in range [0, 11] (both inclusive), deciding which AC move to apply.
            Odd values affect r_1; even values affect r_0.
            The complete mappling between move_id and moves is as below:
            0. r_1 --> r_1 r_0
            1. r_0 --> r_0 r_1^{-1}
            2. r_1 --> r_1 r_0^{-1}
            3. r_0 --> r_0 r_1
            4: r_1 --> x_0^{-1} r_1 x_0
            5: r_0 ---> x_1^{-1} r_0 x_1
            6: r_1 --> x_1^{-1} r_1 x_1
            7: r_0 ---> x_0 r_0 x_0^{-1}
            8: r_1 --> x_0 r_1 x_0^{-1}
            9: r_0 --> x_1 r_0 x_1^{-1}
            10: r_1 --> x_1 r_1 x_1^{-1}
            11: r_0 --> x_0^{-1} r_0 x_0
    presentation: A NumPy Array representation the input presentation.
    max_relator_length: The maximum length a relator is allowed to take.
                        If the application of an AC move results in a relator with length larger than max_relator_length,
                        the original presentation is returned.
    lengths: A list of lengths of words in the presentation.
    cyclical: A bool; whether to cyclically reduce words in the resultant presentation or not.
    """
    assert move_id in range(
        0, 12
    ), f"Expect n to be in range 0-11 (both inclusive); got {move_id}"

    if move_id in range(0, 4):
        move_id += 1
        i = move_id % 2
        j = 1 - i
        sign_parity = ((move_id - i) // 2) % 2
        sign = (-1) ** sign_parity
        move = concatenate_relators
    elif move_id in range(4, 12):
        move_id += 1
        i = move_id % 2
        jp = ((move_id - i) // 2) % 2  # = 0 or 1
        sign_parity = ((move_id - i - 2 * jp) // 4) % 2
        j = jp + 1  # = 1 or 2
        sign = (-1) ** sign_parity
        move = conjugate

    presentation, lengths = move(
        presentation=presentation,
        max_relator_length=max_relator_length,
        i=i,
        j=j,
        sign=sign,
        lengths=lengths,
    )

    # TODO: simplify_presentation seems to do something non-trivial even when
    # cyclical=False. I ran into trouble by putting an `if cyclical==False` cond
    # before the next lines of code.
    # This is confusing because I thought cojugate and concatenate_relators
    # already do the cyclical=False simplification.

    # TODO: cyclical should probably be called cylically_reduce.
    presentation, lengths = simplify_presentation(
        presentation=presentation,
        max_relator_length=max_relator_length,
        lengths_of_words=lengths,
        cyclical=cyclical,
    )

    return presentation, lengths
